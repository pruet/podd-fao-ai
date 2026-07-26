import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import io
import json
import sys
import os

# Ensure Src is in sys.path
sys.path.append(os.path.dirname(__file__))

from main import app, get_genai_client

class TestFAO_PODD_API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    @patch("main.get_genai_client")
    def test_analyze_direct_upload(self, mock_get_client):
        # Mock Gemini Response
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
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
        
        # Mock PIL.Image.open to prevent actual image decoding errors
        with patch("PIL.Image.open") as mock_image_open:
            mock_image_open.return_value = MagicMock()
            
            response = self.client.post(
                "/analyze",
                files={"image": ("test.jpg", image_data, "image/jpeg")},
                data={"lang": "en", "description": "Limping cow"}
            )
            
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["is_valid_animal_image"])
        self.assertEqual(res_json["animal_type"], "Cattle")
        self.assertEqual(res_json["diseases"][0]["name"], "Foot and Mouth Disease")

    @patch("main.get_genai_client")
    @patch("main.get_lahis_token", new_callable=AsyncMock)
    @patch("main.fetch_lahis_report_image", new_callable=AsyncMock)
    @patch.dict(os.environ, {
        "TENANT_API_URL": "https://demo.api.lahis.ohtk.org",
        "LAHIS_CLIENT_ID": "mock_client",
        "LAHIS_CLIENT_SECRET": "mock_secret"
    })
    def test_analyze_lahis_report_id(self, mock_fetch_image, mock_get_token, mock_get_client):
        # Mock token & image retrieval
        mock_get_token.return_value = "mock_access_token"
        mock_fetch_image.return_value = b"mock image bytes"
        
        # Mock Gemini Response
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
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
        
        # Mock PIL.Image.open
        with patch("PIL.Image.open") as mock_image_open:
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
        mock_fetch_image.assert_called_once_with("12345", "mock_access_token", "https://demo.api.lahis.ohtk.org")

    def test_analyze_missing_parameters(self):
        # Call without image or report_id
        response = self.client.post(
            "/analyze",
            data={"lang": "en"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Either image file or report_id must be provided.", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
