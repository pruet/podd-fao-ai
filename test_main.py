import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import io
import sqlite3
import json
import sys
import os

# Ensure Src is in sys.path
sys.path.append(os.path.dirname(__file__))

from main import app, get_genai_client, init_db, DB_PATH

class TestFAO_PODD_API(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Guarantee database exists
        init_db()
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute("DELETE FROM cluster_reports")
            conn.commit()
            conn.close()
        except Exception:
            pass
        self.client = TestClient(app)
        
    @patch("main.get_genai_client")
    def test_analyze_direct_upload(self, mock_get_client):
        # Mock Gemini Response
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": [
                {
                    "name": "Foot and Mouth Disease",
                    "confidence": 0.9,
                    "reasoning": "Classic oral lesions"
                }
            ]
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_genai_client
        
        # Test image file
        image_data = io.BytesIO(b"fake image data")
        
        # Mock main.Image.open to prevent actual image decoding errors
        with patch("main.Image.open") as mock_image_open:
            mock_image_open.return_value = MagicMock()
            
            response = self.client.post(
                "/analyze",
                files={"images": ("test.jpg", image_data, "image/jpeg")},
                data={"lang": "en", "description": "Limping cow"}
            )
            
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["is_valid_animal_image"])
        self.assertEqual(res_json["animal_type"], "Cattle")
        self.assertEqual(res_json["diseases"][0]["name"], "Foot and Mouth Disease")
        self.assertIn("basic_response", res_json["diseases"][0])
        self.assertIsInstance(res_json["diseases"][0]["basic_response"], list)
        self.assertTrue(len(res_json["diseases"][0]["basic_response"]) > 0)

    @patch("main.get_genai_client")
    @patch("main.get_lahis_token", new_callable=AsyncMock)
    @patch("main.fetch_lahis_report_images", new_callable=AsyncMock)
    @patch.dict(os.environ, {
        "TENANT_API_URL": "https://demo.api.lahis.ohtk.org",
        "LAHIS_CLIENT_ID": "mock_client",
        "LAHIS_CLIENT_SECRET": "mock_secret"
    })
    def test_analyze_lahis_report_id(self, mock_fetch_images, mock_get_token, mock_get_client):
        # Mock token & image retrieval
        mock_get_token.return_value = "mock_access_token"
        mock_fetch_images.return_value = [b"mock image bytes"]
        
        # Mock Gemini Response
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Pig",
            "diseases": [
                {
                    "name": "African Swine Fever",
                    "confidence": 0.8,
                    "reasoning": "Hemorrhagic spots"
                }
            ]
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_genai_client
        
        # Mock main.Image.open
        with patch("main.Image.open") as mock_image_open:
            mock_image_open.return_value = MagicMock()
            
            response = self.client.post(
                "/analyze",
                data={"report_id": "12345", "lang": "en"}
            )
            
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["is_valid_animal_image"])
        self.assertEqual(res_json["animal_type"], "Pig")
        self.assertEqual(res_json["diseases"][0]["name"], "African Swine Fever")
        
        # Verify LAHIS functions were called
        mock_get_token.assert_called_once_with("mock_client", "mock_secret", "https://demo.api.lahis.ohtk.org")
        mock_fetch_images.assert_called_once_with("12345", "mock_access_token", "https://demo.api.lahis.ohtk.org")

    def test_analyze_missing_parameters(self):
        # Call without image or report_id
        response = self.client.post(
            "/analyze",
            data={"lang": "en"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Either image files or report_id must be provided.", response.json()["detail"])

    @patch("httpx.AsyncClient")
    async def test_get_lahis_token_logic(self, mock_client_class):
        from main import get_lahis_token
        # Mock client instance and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "correct_token"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Setup context manager return
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        token = await get_lahis_token("my_client", "my_secret", "https://demo.api.lahis.ohtk.org")
        self.assertEqual(token, "correct_token")
        
        # Verify the post endpoint called is the new one
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://demo.api.lahis.ohtk.org/o/token/")
        self.assertEqual(kwargs["data"]["client_id"], "my_client")

    @patch("httpx.AsyncClient")
    async def test_fetch_lahis_report_images_logic(self, mock_client_class):
        from main import fetch_lahis_report_images
        mock_client = MagicMock()
        
        # Mock responses
        mock_report_response = MagicMock()
        mock_report_response.json.return_value = {
            "images": [
                {
                    "id": "img123",
                    "links": {
                        "content": "/api/integrations/v1/reports/123/images/img123/content"
                    }
                }
            ]
        }
        mock_report_response.raise_for_status = MagicMock()
        
        mock_img_response = MagicMock()
        mock_img_response.content = b"fake-downloaded-bytes"
        mock_img_response.raise_for_status = MagicMock()
        
        mock_client.get = AsyncMock(side_effect=[mock_report_response, mock_img_response])
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        content = await fetch_lahis_report_images("123", "tokenabc", "https://demo.api.lahis.ohtk.org")
        self.assertEqual(content, [b"fake-downloaded-bytes"])
        
        # Verify GET calls
        self.assertEqual(mock_client.get.call_count, 2)
        
        # First call: get report metadata
        first_call = mock_client.get.call_args_list[0]
        self.assertEqual(first_call[0][0], "https://demo.api.lahis.ohtk.org/api/integrations/v1/reports/123/images")
        self.assertEqual(first_call[1]["headers"]["Authorization"], "Bearer tokenabc")
        
        # Second call: download image content
        second_call = mock_client.get.call_args_list[1]
        self.assertEqual(second_call[0][0], "https://demo.api.lahis.ohtk.org/api/integrations/v1/reports/123/images/img123/content")

    def test_dashboard_and_logs_endpoints(self):
        # 1. Test unauthorized access (should return 401)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 401)
        
        response = self.client.get("/api/logs")
        self.assertEqual(response.status_code, 401)

        # 2. Test authorized access (should return 200)
        # We pass auth=("admin", "PoddFaoSecure2026!") which corresponds to local default settings in setUp/env
        response = self.client.get("/", auth=("admin", "PoddFaoSecure2026!"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("FAO-PODD Diagnostics Dashboard", response.text)

        response = self.client.get("/api/logs", auth=("admin", "PoddFaoSecure2026!"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers["content-type"])
        self.toBeInstance = isinstance(response.json(), list)
        self.assertTrue(self.toBeInstance)

    @patch("main.get_genai_client")
    @patch("main.get_lahis_token", new_callable=AsyncMock)
    @patch("main.fetch_lahis_report_images", new_callable=AsyncMock)
    @patch("main.submit_lahis_comment", new_callable=AsyncMock)
    @patch.dict(os.environ, {
        "TENANT_API_URL": "https://demo.api.lahis.ohtk.org",
        "LAHIS_CLIENT_ID": "mock_client",
        "LAHIS_CLIENT_SECRET": "mock_secret",
        "LAHIS_SIGNING_SECRET": ""
    })
    def test_analyze_webhook_json(self, mock_submit_comment, mock_fetch_images, mock_get_token, mock_get_client):
        mock_get_token.return_value = "mock_access_token"
        mock_fetch_images.return_value = [b"mock image bytes"]
        
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Goat",
            "diseases": [
                {
                    "name": "Peste des Petits Ruminants",
                    "confidence": 0.85,
                    "reasoning": "High fever, nasal discharge"
                }
            ]
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_genai_client
        
        payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "event-123-abc",
            "producedAt": "2026-07-21T10:30:00+00:00",
            "tenant": {"schema": "demo", "code": "demo", "name": "LAHIS Demo"},
            "report": {
                "id": "report-999",
                "createdAt": "2026-07-21T10:29:58+00:00",
                "incidentDate": "2026-07-21",
                "reportType": {"id": "type-123", "name": "Animal Sick/Death", "category": "Animal"},
                "relevantAuthorityIds": [12],
                "caseId": None
            },
            "links": {
                "incident": "/api/integrations/v1/incidents/report-999",
                "comments": "/api/integrations/v1/reports/report-999/comments",
                "riskAssessments": "/api/integrations/v1/reports/report-999/risk-assessments",
                "images": "/api/integrations/v1/reports/report-999/images"
            }
        }
        
        with patch("main.Image.open") as mock_image_open:
            mock_image_open.return_value = MagicMock()
            
            response = self.client.post(
                "/analyze",
                json=payload
            )
            
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["is_valid_animal_image"])
        self.assertEqual(res_json["animal_type"], "Goat")
        
        mock_submit_comment.assert_called_once()
        args, kwargs = mock_submit_comment.call_args
        self.assertEqual(args[0], "report-999")
        self.assertEqual(args[1], "event-123-abc")
        self.assertIn("Peste des Petits Ruminants", args[2])
        self.assertEqual(args[3], 0.85)

    @patch("main.get_genai_client")
    @patch.dict(os.environ, {
        "LAHIS_SIGNING_SECRET": "test_signing_secret"
    })
    def test_webhook_invalid_signature_logged(self, mock_get_client):
        response = self.client.post(
            "/analyze",
            json={"eventType": "report.submitted"}
        )
        self.assertEqual(response.status_code, 401)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM api_logs WHERE status_code = 401 ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[4], 401) # status_code

    @patch("main.get_genai_client")
    @patch("main.get_lahis_token", new_callable=AsyncMock)
    @patch("main.fetch_lahis_report_images", new_callable=AsyncMock)
    @patch("main.submit_lahis_comment", new_callable=AsyncMock)
    @patch.dict(os.environ, {
        "TENANT_API_URL": "https://demo.api.lahis.ohtk.org",
        "LAHIS_CLIENT_ID": "mock_client",
        "LAHIS_CLIENT_SECRET": "mock_secret",
        "LAHIS_SIGNING_SECRET": "my_secret_key"
    })
    def test_webhook_signature_with_trailing_slash(self, mock_submit_comment, mock_fetch_images, mock_get_token, mock_get_client):
        import hmac
        import hashlib
        
        mock_get_token.return_value = "mock_access_token"
        mock_fetch_images.return_value = [b"mock image bytes"]
        
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Goat",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_genai_client
        
        payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "event-123-abc",
            "producedAt": "2026-07-21T10:30:00+00:00",
            "tenant": {"schema": "demo", "code": "demo", "name": "LAHIS Demo"},
            "report": {
                "id": "report-999",
                "createdAt": "2026-07-21T10:29:58+00:00",
                "incidentDate": "2026-07-21",
                "reportType": {"id": "type-123", "name": "Animal Sick/Death", "category": "Animal"},
                "relevantAuthorityIds": [12],
                "caseId": None
            },
            "links": {
                "incident": "/api/integrations/v1/incidents/report-999",
                "comments": "/api/integrations/v1/reports/report-999/comments",
                "riskAssessments": "/api/integrations/v1/reports/report-999/risk-assessments",
                "images": "/api/integrations/v1/reports/report-999/images"
            }
        }
        
        raw_body = json.dumps(payload).encode("utf-8")
        timestamp = "2026-07-21T10:30:00Z"
        
        message = b"POST\n/analyze/\n" + timestamp.encode("utf-8") + b"\n" + raw_body
        signature = hmac.new(b"my_secret_key", message, hashlib.sha256).hexdigest()
        
        with patch("main.Image.open") as mock_image_open:
            mock_image_open.return_value = MagicMock()
            
            response = self.client.post(
                "/analyze",
                content=raw_body,
                headers={
                    "Content-Type": "application/json",
                    "X-OHTK-Timestamp": timestamp,
                    "X-OHTK-Signature": signature
                }
            )
            
        self.assertEqual(response.status_code, 200)

    @patch("main.get_genai_client")
    def test_image_logging_and_serving(self, mock_get_client):
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_genai_client

        image_data = io.BytesIO(b"fake image data")
        with patch("main.Image.open") as mock_image_open:
            mock_img = MagicMock()
            mock_img.format = "PNG"
            mock_image_open.return_value = mock_img
            
            response = self.client.post(
                "/analyze",
                files={"images": ("test.png", image_data, "image/png")},
                data={"lang": "en", "description": "test image logging"}
            )
            
        self.assertEqual(response.status_code, 200)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM api_logs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        saved_path = row[0]
        self.assertIsNotNone(saved_path)
        self.assertTrue(saved_path.startswith("logs/images/"))
        
        src_dir = os.path.dirname(__file__)
        full_image_path = os.path.join(src_dir, saved_path)
        self.assertTrue(os.path.exists(full_image_path))
        
        filename = saved_path.split("/")[-1]
        auth_headers = {"Authorization": "Basic YWRtaW46UG9kZEZhb1NlY3VyZTIwMjYh"}
        image_response = self.client.get(f"/api/logs/image/{filename}", headers=auth_headers)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.content, b"fake image data")
        
        try:
            os.remove(full_image_path)
        except Exception:
            pass

    @patch("main.get_genai_client")
    def test_multiple_images_upload(self, mock_get_client):
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_genai_client

        image_data_1 = io.BytesIO(b"fake image data 1")
        image_data_2 = io.BytesIO(b"fake image data 2")
        with patch("main.Image.open") as mock_image_open:
            mock_img = MagicMock()
            mock_img.format = "PNG"
            mock_image_open.return_value = mock_img
            
            response = self.client.post(
                "/analyze",
                files=[
                    ("images", ("test1.png", image_data_1, "image/png")),
                    ("images", ("test2.png", image_data_2, "image/png"))
                ],
                data={"lang": "en", "description": "test multiple images"}
            )
            
        self.assertEqual(response.status_code, 200)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM api_logs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        saved_path = row[0]
        self.assertIsNotNone(saved_path)
        # Verify it contains comma-separated paths
        paths = saved_path.split(",")
        self.assertEqual(len(paths), 2)
        
        # Clean up files
        src_dir = os.path.dirname(__file__)
        for p in paths:
            try:
                os.remove(os.path.join(src_dir, p))
            except Exception:
                pass

    def test_detect_cluster_missing_report_id(self):
        response = self.client.post("/detect-cluster", data={"lang": "en"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("report_id is required", response.json()["detail"])

    @patch("main.submit_lahis_cluster", new_callable=AsyncMock)
    @patch("main.fetch_lahis_incident_detail", new_callable=AsyncMock)
    @patch("main.get_lahis_token", new_callable=AsyncMock)
    @patch.dict(os.environ, {
        "TENANT_API_URL": "https://demo.api.lahis.ohtk.org",
        "LAHIS_CLIENT_ID": "mock_client",
        "LAHIS_CLIENT_SECRET": "mock_secret"
    })
    def test_detect_cluster_threshold_and_callback(self, mock_get_token, mock_fetch_detail, mock_submit_cluster):
        mock_get_token.return_value = "mock_token"
        mock_submit_cluster.return_value = {"status": "accepted", "id": "cluster-111"}

        # Simulate reports spanning > 14 days for village 99
        # Report 1: Day 1 (2026-08-01)
        # Report 2: Day 10 (2026-08-10) -> span 9 days (<= 14 days)
        # Report 3: Day 17 (2026-08-17) -> span 16 days (> 14 days)
        mock_fetch_detail.side_effect = [
            {"incident": {"id": "rep-001", "incidentDate": "2026-08-01", "village": {"id": 99, "name": "Ban Na"}}},
            {"incident": {"id": "rep-002", "incidentDate": "2026-08-10", "village": {"id": 99, "name": "Ban Na"}}},
            {"incident": {"id": "rep-003", "incidentDate": "2026-08-17", "village": {"id": 99, "name": "Ban Na"}}}
        ]

        # 1st request -> span 0 days, cluster_detected = False
        res1 = self.client.post("/detect-cluster", data={"report_id": "rep-001"})
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertFalse(data1["cluster_detected"])
        self.assertEqual(data1["span_days"], 0)
        mock_submit_cluster.assert_not_called()

        # 2nd request -> span 9 days, cluster_detected = False
        res2 = self.client.post("/detect-cluster", data={"report_id": "rep-002"})
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertFalse(data2["cluster_detected"])
        self.assertEqual(data2["span_days"], 9)
        mock_submit_cluster.assert_not_called()

        # 3rd request -> span 16 days, cluster_detected = True (> 14 days)
        res3 = self.client.post("/detect-cluster", data={"report_id": "rep-003"})
        self.assertEqual(res3.status_code, 200)
        data3 = res3.json()
        self.assertTrue(data3["cluster_detected"])
        self.assertEqual(data3["span_days"], 16)
        mock_submit_cluster.assert_called_once()

    def test_get_clusters_api(self):
        auth_headers = {"Authorization": "Basic YWRtaW46UG9kZEZhb1NlY3VyZTIwMjYh"}
        response = self.client.get("/api/clusters", headers=auth_headers)
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertIn("stats", res_json)
        self.assertIn("clusters", res_json)

    def test_get_diseases_api(self):
        response = self.client.get("/api/diseases")
        self.assertEqual(response.status_code, 200)
        diseases = response.json()
        self.assertIn("Chicken", diseases)
        self.assertIn("Pig", diseases)
        chicken_disease_names = [d["name"] if isinstance(d, dict) else d for d in diseases["Chicken"]]
        self.assertIn("Avian influenza", chicken_disease_names)
        if isinstance(diseases["Chicken"][0], dict):
            self.assertIn("symptoms", diseases["Chicken"][0])
            self.assertIsInstance(diseases["Chicken"][0]["symptoms"], list)
            self.assertIn("basic_response", diseases["Chicken"][0])
            basic_resp = diseases["Chicken"][0]["basic_response"]
            self.assertIn("lo", basic_resp)
            self.assertIn("th", basic_resp)
            self.assertIn("en", basic_resp)
            self.assertIsInstance(basic_resp["lo"], list)
            self.assertIsInstance(basic_resp["th"], list)
            self.assertIsInstance(basic_resp["en"], list)

if __name__ == "__main__":
    unittest.main()


