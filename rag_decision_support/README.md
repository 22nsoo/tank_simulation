# TankChallenge RAG Decision Support

TankChallenge shot logs and live LiDAR/YOLO battlefield context are used as RAG evidence for fire-control support.

The system retrieves similar historical shot cases, optionally retrieves the latest live tactical snapshot from ChromaDB, and uses an LLM to produce FIRE/HOLD recommendations and Korean chatbot answers. If the LLM is unavailable, it falls back to the deterministic rule-based recommender.

## What It Does

- Builds shot-case documents from `shot_analysis/shot_log_*.csv`
- Builds moving-target shot cases from `shot_analysis/moving_target_logs/*.csv`
- Searches similar historical cases with FAISS, ChromaDB, or numeric/text hybrid search
- Uses `tank_rag_llm.py` for LLM-based FIRE/HOLD decisions
- Provides a dashboard and JSON APIs from `app.py`
- Adds a RAG chat panel for Korean tactical Q&A
- Reads live LiDAR/YOLO server state from `http://127.0.0.1:5000`
- Stores the latest battlefield snapshot in ChromaDB collection `tank_tactical_context`
- Uses live tactical context for enemy count, nearest enemy, obstacle, and line-of-fire questions
- Produces a high-level drive/fire action recommendation from RAG + live tactical context
- Runs a local drive planner for movement actions, including waypoint following, PD steering, clearance stop, TTC stop, and replan signaling

## Main Files

```text
rag_decision_support/
  app.py                    Flask dashboard and API
  tank_rag.py               Base RAG search and rule recommender
  tank_rag_llm.py           LLM decision mode and chatbot prompts
  tactical_context.py       Live LiDAR/YOLO tactical context -> ChromaDB
  drive_planner.py          Local waypoint/TTC/clearance drive planner
  world_map.py              300x300 .map parser for dashboard rendering
  pca_visualize.py          Embedding PCA visualization
  case_index.jsonl          Shot-case JSONL index
  faiss_cases.index         FAISS vector index
  faiss_cases_meta.json     FAISS metadata and case payloads
  chroma_store/             ChromaDB persistent storage
  templates/index.html      Dashboard UI
  static/styles.css         Dashboard styling
```

## Data Flow

```text
Shot CSV logs
  -> ShotCase documents
  -> SentenceTransformer embeddings
  -> FAISS / ChromaDB historical retrieval

Live lidar_cluster.py server on port 5000
  -> fire_status / lidar_status / vision_status / fusion_status / aim_status
  -> tactical context summary
  -> ChromaDB collection: tank_tactical_context

Current query + historical cases + live tactical context
  -> LLM recommendation or Korean chatbot answer
  -> drive/fire action recommendation
  -> local drive planner command for movement actions
  -> rule fallback if LLM/API key is unavailable
```

## Install

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Main dependencies:

```text
flask
sentence-transformers
faiss-cpu
chromadb
google-generativeai
matplotlib
scikit-learn
```

## Build Indexes

Run from the project root:

```powershell
python rag_decision_support\tank_rag.py build
python rag_decision_support\tank_rag.py build-embeddings
python rag_decision_support\tank_rag.py build-chroma
python rag_decision_support\pca_visualize.py
```

## Run Dashboard

```powershell
python rag_decision_support\app.py
```

Open:

```text
http://127.0.0.1:5056
```

The dashboard includes:

- Current situation input form
- FIRE/HOLD recommendation
- Drive/fire action card with safe movement target
- Local drive planner card with command, clearance, TTC, and replan state
- 300x300 world-map panel with static obstacles, tank pose, safe target, and waypoints
- Tactical top-down map
- LiDAR body-relative polar view
- RAG chat
- Retrieval quality comparison
- Retrieved similar cases
- PCA embedding map

## LLM Setup

Default mode: external LLM calls are allowed when `LLAMA_API_KEY` is set.

To block paid LLM calls explicitly:

```powershell
$env:TANK_RAG_ALLOW_PAID_LLM="false"
python rag_decision_support\app.py
```

Llama API mode:

```powershell
$env:LLAMA_API_KEY="YOUR_LLAMA_API_KEY"
python rag_decision_support\app.py
```

Optional Llama settings:

```text
TANK_RAG_LLM_PROVIDER=llama
TANK_RAG_LLAMA_MODEL=llama3.1-70b
LLAMA_API_BASE_URL=https://api.llama-api.com
```

Gemini is still available as an optional fallback provider:

```powershell
$env:TANK_RAG_LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="YOUR_KEY"
python rag_decision_support\app.py
```

OpenAI support has been removed from this RAG app. If the selected LLM key, package, or provider call fails, the app returns a rule-based fallback result instead of crashing.

## Live LiDAR/YOLO Tactical Context

The RAG app can read live state from the integrated TankChallenge LiDAR/YOLO server in this folder.

Expected live source:

```text
http://127.0.0.1:5000
```

The RAG app tries to read:

```text
/fire_status
/lidar_status
/vision_status
/fusion_status
/aim_status
/action_debug
```

It summarizes and stores:

- Enemy-like object count
- Nearest enemy label, distance, and angle
- Obstacle cluster count
- Front-lane obstacle list
- Whether the front lane appears blocked
- Current fire/aim status fields

The latest summary is upserted into ChromaDB:

```text
collection: tank_tactical_context
id: live_latest
```

If the 5000 server is off, tactical context returns `unavailable` and the app continues with historical shot-case RAG.

For actual simulator actuation, run the integrated LiDAR/YOLO control server:

```powershell
python rag_decision_support\lidar_cluster.py
```

That server owns `/get_action`. It combines the LiDAR/YOLO perception stream with local waypoint/TTC/clearance driving through `rag_decision_support/integrated_control.py` and `shot_analysis/fire_logic.py` aiming/fire control. The RAG dashboard remains the decision/monitoring UI and reads the 5000 server state.

## Tactical Context APIs

Ingest latest live context:

```text
GET /api/tactical/ingest
POST /api/tactical/ingest
```

Search tactical context:

```text
GET /api/tactical/search?q=front blocked
GET /api/tactical/search?q=enemy count
```

Override live source:

```text
GET /api/tactical/ingest?source_base=http://127.0.0.1:5000
```

World map:

```text
GET /api/world-map
GET /api/world-map?map_file=NewMap_120m_to_40m_4m_varied.map
```

The dashboard reads `.map` JSON files from the project root and renders them as a 300m x 300m world canvas. The panel overlays static map obstacles, the live player pose when available, the safe movement destination, and local planner waypoints.

## Query APIs

Recommendation:

```text
GET /api/query?backend=faiss&distance=85&body_error=3&turret_error=0.8&pitch_error=-0.1&enemy_speed=0.4
```

Chat:

```text
POST /api/chat
Content-Type: application/json

{
  "message": "주변에 적 몇 명이야? 정면에 장애물 있어?",
  "backend": "faiss",
  "distance": 85,
  "body_error": 3,
  "turret_error": 0.8,
  "pitch_error": -0.1,
  "enemy_speed": 0.4,
  "top_k": 5
}
```

Drive/fire action:

```text
GET /api/action?backend=faiss&distance=85&body_error=3&turret_error=0.8&pitch_error=-0.1&enemy_speed=0.4
POST /api/action
```

Example action response:

```json
{
  "action": {
    "action": "MOVE_LEFT",
    "confidence": 0.72,
    "reason": "Front lane is blocked; left side appears less blocked than right side.",
    "control_hint": {
      "move": "FORWARD",
      "turn": "LEFT",
      "fire": false
    },
    "safe_destination": {
      "relative_angle_deg": -45.0,
      "relative_distance_m": 12.0,
      "world": {
        "x": 121.3,
        "y": 0.0,
        "z": 203.6
      }
    },
    "enemy_count": 2,
    "front_blocked": true
  },
  "drive_plan": {
    "available": true,
    "control_mode": "path_follow",
    "reason": "following local waypoints toward safe destination",
    "command": {
      "move_ws": "W",
      "move_ad": "A",
      "move_weight": 0.36,
      "turn_weight": 0.42,
      "wp_index": 0,
      "heading_error": -18.5,
      "target_yaw": 312.0,
      "distance_to_wp_m": 6.0
    },
    "waypoints": [
      {"x": 118.2, "y": 0.0, "z": 206.4}
    ],
    "goal": {"x": 121.3, "y": 0.0, "z": 203.6},
    "min_clearance_m": 8.4,
    "min_ttc_s": 3.2,
    "replan_requested": false
  }
}
```

Possible high-level actions:

```text
FIRE       stop and fire
HOLD_AIM   stop, align turret/pitch, do not fire yet
MOVE_LEFT  move forward while steering left around obstacle
MOVE_RIGHT move forward while steering right around obstacle
REVERSE    back up because a front obstacle is too close
SCAN       no confirmed enemy; scan cautiously
REPLAN     route-level replanning requested by local planner risk checks
```

`safe_destination.world` is available only when the live LiDAR/YOLO payload exposes player position and body yaw, such as `playerPos` and `playerBodyX`. If those fields are not available, the response still includes body-relative movement intent through `relative_angle_deg` and `relative_distance_m`.

## Local Drive Planner

`drive_planner.py` is the RAG-side local movement planner adapted from the Tank Challenge driving module. It is intentionally output-only: it calculates simulator-style commands but does not send them to the tank by itself.

Planner behavior:

- Builds short waypoints from the current player pose to `safe_destination`
- Selects a lookahead waypoint
- Computes heading error and PD-style turn weight
- Emits `move_ws`, `move_ad`, `move_weight`, and `turn_weight`
- Stops on emergency positional clearance
- Stops on low TTC
- Marks `replan_requested` when the planned corridor is unsafe

Common `drive_plan.control_mode` values:

```text
path_follow             follow generated local waypoints
path_follow_replan_ahead follow for now, but request route replanning
clearance_stop          stop because an obstacle is too close
ttc_stop                stop because time-to-collision is too low
corridor_replan_stop    stop and replan because the corridor is blocked nearby
stationary_action       FIRE/HOLD_AIM/SCAN does not need waypoint following
no_pose                 live player position or body yaw is unavailable
no_goal                 safe destination is unavailable
```

Live query bridge:

```text
GET /api/live-query?source=http://127.0.0.1:5000/fire_status&backend=faiss
```

## CLI Examples

Rule-based or vector search query:

```powershell
python rag_decision_support\tank_rag.py query --backend faiss --distance 85 --body-error 3 --turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4
```

LLM decision query:

```powershell
python rag_decision_support\tank_rag_llm.py query --backend faiss --decision-mode llm --distance 85 --body-error 3 --turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4
```

Generate report:

```powershell
python rag_decision_support\tank_rag.py report --backend faiss
```

## Notes

- FAISS is the main fast local vector-search backend for shot cases.
- ChromaDB is used both for shot-case vector storage and live tactical context storage.
- The tactical context currently keeps a compact latest snapshot as `live_latest`; this avoids filling the DB with every LiDAR frame.
- For richer long-term memory, add event snapshots only on meaningful changes such as enemy detected, target lock, obstacle block, shot fired, hit, or miss.
