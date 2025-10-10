```mermaid
flowchart LR
  %% ===== System Architecture (runtime dataflow) =====
  subgraph Inputs_Outputs["Inputs / Outputs"]
    Cam["Camera"]
    User["User Command"]
    Servos["3x Servos (Omniverse 3-Wheel Base)"]
  end

  subgraph Core["VisionNavBot Core"]
    subgraph IO["IO Layer"]
      IOCam["io_camera.py (grab, undistort)"]
      IORobot["io_robot.py (PWM, GPIO)"]
    end

    subgraph Perception["Perception"]
      SLAM["slam.py Mono SLAM/VIO pose (x,y,θ) + scale"]
      DET["perception.py Object+Color Detect"]
      FREE["perception.py Depth/Freespace"]
    end

    subgraph Mapping["Mapping"]
      MAP["mapping.py Costmap Builder"]
      SEM["mapping.py Semantic Landmarks (object, color, pose)"]
      PLACES["mapping.py Named Places"]
    end

    subgraph Planning["Planning"]
      GLB["planner.py Global Plan (A*/Graph)"]
      LOC["planner.py Local Control (Pure-Pursuit)"]
      SRCH["planner.py Search Behavior"]
    end

    subgraph Control["Control"]
      CTR["controller.py v, ω → PWM TTC Safety Brake"]
    end

    subgraph Lang["Language Grounding"]
      LPARSE["language.py Parse goal (object/color/place)"]
    end
  end

  %% Edges
  Cam --> IOCam
  IOCam --> SLAM
  IOCam --> DET
  IOCam --> FREE

  SLAM --> MAP
  FREE --> MAP
  DET --> SEM
  SLAM -->|pose| SEM

  MAP --> GLB
  SEM --> GLB
  PLACES --> GLB

  User --> LPARSE
  LPARSE -->|goal| GLB
  SLAM -->|pose| GLB

  GLB --> LOC
  MAP --> LOC
  SLAM --> LOC

  LOC --> CTR
  CTR --> IORobot
  IORobot --> Servos
```



```mermaid
sequenceDiagram
  %% ===== Command-to-Action Loop =====
  participant U as User
  participant L as language.py
  participant S as slam.py
  participant P as perception.py
  participant M as mapping.py
  participant G as planner.py
  participant C as controller.py
  participant R as io_robot.py

  U->>L: "move to black chair/ defined coordinates"
  L-->>G: goal = {cls:"chair", color:"black"} or place

  loop Control Loop (10–20 Hz)
    P->>P: detect objects & freespace
    P-->>M: detections, freespace
    S-->>M: pose (x,y,θ)
    M-->>G: costmap, landmarks, places
    S-->>G: current pose

    alt have goal pose
      G-->>G: global path (A*) + local target
    else need search
      G-->>G: spin/explore to locate target
    end

    G-->>C: (vx, vy, ω)
    C-->>R: PWM (wheel1, wheel2, wheel3)
  end

```
