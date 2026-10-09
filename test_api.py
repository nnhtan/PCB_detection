import io
import json
from backendweb import app

def test_backend():
    client = app.test_client()

    print("1. Testing GET / ...")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    print("   -> OK (HTML served, length:", len(res.data), ")")

    print("2. Testing GET /api/info ...")
    res = client.get('/api/info')
    assert res.status_code == 200
    info = res.get_json()
    print("   -> OK:", info["model_name"], "| Device:", info["device"], "| Classes:", info["total_classes"])

    print("3. Testing GET /api/reference ...")
    res = client.get('/api/reference')
    assert res.status_code == 200
    ref = res.get_json()
    print("   -> OK: Reference components count =", ref["components_count"])

    print("4. Testing GET /api/sample ...")
    res = client.get('/api/sample')
    assert res.status_code == 200
    sample = res.get_json()
    print("   -> OK: Sample image loaded, dimensions:", sample["width"], "x", sample["height"])

    print("5. Testing POST /api/detect with 2.jpg ...")
    with open('2.jpg', 'rb') as f:
        img_bytes = f.read()
    res = client.post('/api/detect', data={'image': (io.BytesIO(img_bytes), '2.jpg')})
    assert res.status_code == 200
    det = res.get_json()
    print("   -> OK: Total detections =", det["total_count"], "| Latency =", det["latency_ms"], "ms")

    print("6. Testing POST /api/inspect with 2.jpg ...")
    res = client.post('/api/inspect', data={'image': (io.BytesIO(img_bytes), '2.jpg'), 'conf': '0.25'})
    assert res.status_code == 200
    insp = res.get_json()
    print("   -> OK: Verdict =", insp["verdict"], "| Matched =", insp["stats"]["matched"], "| Missing =", insp["stats"]["missing"], "| Sai khác =", insp["stats"]["extra"], "| Latency =", insp["latency_ms"], "ms")

    print("7. Testing GET /api/camera/status ...")
    res = client.get('/api/camera/status')
    assert res.status_code == 200
    cam_status = res.get_json()
    print("   -> OK: Camera status =", cam_status["camera"])

    print("\nALL 7 TESTS PASSED SUCCESSFULLY! 🎉")

if __name__ == '__main__':
    test_backend()
