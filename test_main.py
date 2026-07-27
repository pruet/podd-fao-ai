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

class TestFAO_PODD_API(unittest.IsolatedAsyncioTestCase):
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
    async def test_fetch_lahis_report_image_logic(self, mock_client_class):
        from main import fetch_lahis_report_image
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
        
        content = await fetch_lahis_report_image("123", "tokenabc", "https://demo.api.lahis.ohtk.org")
        self.assertEqual(content, b"fake-downloaded-bytes")
        
        # Verify GET calls
        self.assertEqual(mock_client.get.call_count, 2)
        
        # First call: get report metadata
        first_call = mock_client.get.call_args_list[0]
        self.assertEqual(first_call[0][0], "https://demo.api.lahis.ohtk.org/api/integrations/v1/reports/123/images")
        self.assertEqual(first_call[1]["headers"]["Authorization"], "Bearer tokenabc")
        
        # Second call: download image content
        second_call = mock_client.get.call_args_list[1]
        self.assertEqual(second_call[0][0], "https://demo.api.lahis.ohtk.org/api/integrations/v1/reports/123/images/img123/content")

if __name__ == "__main__":
    unittest.main()
