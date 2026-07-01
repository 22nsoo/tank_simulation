# Tank Challenge

TankChallenge experiment workspace with LiDAR/YOLO perception, fire-control logic, shot analysis, and RAG decision support.

## Key Areas

```text
hillmapfire/              LiDAR + YOLO fusion server and fire logic experiments
shot_analysis/            Shot logs, moving-target analysis, plotting scripts
rag_decision_support/     RAG dashboard, LLM decision mode, chatbot, ChromaDB context
misc archive folder       Archived or alternate server/control modules
```

## RAG Dashboard

The current RAG system is documented in:

```text
rag_decision_support/README.md
```

Run:

```powershell
python rag_decision_support\app.py
```

Open:

```text
http://127.0.0.1:5056
```

The dashboard now supports:

- FAISS/Chroma historical shot-case retrieval
- LLM FIRE/HOLD decision mode
- Korean RAG chatbot
- Live LiDAR/YOLO tactical context from `http://127.0.0.1:5000`
- ChromaDB storage for latest tactical context

## LiDAR/YOLO Source

The active perception/control server now lives in the RAG folder:

```powershell
python rag_decision_support\lidar_cluster.py
```

Expected source for the simulator and the RAG dashboard:

```text
http://127.0.0.1:5000
```

This server now combines:

- LiDAR/YOLO perception from `rag_decision_support/lidar_cluster.py`
- local-planner style waypoint/TTC/clearance driving through `rag_decision_support/integrated_control.py`
- ballistic aiming and fire permission through `shot_analysis/fire_logic.py`

Useful control endpoints:

```text
GET  /control_status
POST /control_update
POST /set_destination
POST /get_action
GET  /fire_targets?includeLidarFallback=true
```

Example destination:

```powershell
$body = @{x=80; y=8; z=60} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/set_destination -Method POST -ContentType "application/json" -Body $body
```

If the perception/control server is not running, the RAG app still works with historical shot cases and reports tactical context as unavailable.
