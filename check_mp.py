import mediapipe as mp
print("version:", mp.__version__)
print("has solutions:", hasattr(mp, 'solutions'))
try:
    fm = mp.solutions.face_mesh
    print("solutions.face_mesh OK")
except Exception as e:
    print("solutions error:", e)

try:
    from mediapipe.tasks import python as mp_tasks
    print("tasks API available")
except Exception as e:
    print("tasks error:", e)
