"""Offline catalog tests. No external API keys or network access required."""
import unittest
from unittest.mock import patch

from council import model_catalog as catalog


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        catalog._cache.clear()

    async def test_key_missing_returns_explicit_status(self):
        with patch.object(catalog, '_provider_key', return_value=None):
            result = await catalog.list_provider_models('gemini')
        self.assertEqual(result['status'], 'missing_key')
        self.assertEqual(result['models'], [])

    async def test_invalid_provider_rejected(self):
        with self.assertRaises(ValueError):
            await catalog.list_provider_models('other')

    async def test_openai_response_is_not_claimed_chat_compatible(self):
        with patch.object(catalog, '_request_json', return_value={
            'data': [{'id': 'gpt-sample', 'owned_by': 'openai'},
                     {'id': 'not-a-chat-model'},
                     {'id': ''}]}):
            result = catalog._openai_models('fake')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['compatibility'], 'unknown')
        self.assertIn('не гарантирована', result[0]['note'])

    async def test_gemini_filters_methods_and_preserves_metadata(self):
        with patch.object(catalog, '_request_json', return_value={
            'models': [
                {'name': 'models/gemini-3.8-flash',
                 'displayName': 'Gemini 3.8 Flash',
                 'supportedGenerationMethods': ['generateContent'],
                 'inputTokenLimit': 1048576},
                {'name': 'models/embedding-001',
                 'supportedGenerationMethods': ['embedContent']},
                {'name': 'models/gemini-4-argon',
                 'supportedGenerationMethods': ['generateContent']},
            ]}):
            result = catalog._gemini_models('fake')
        self.assertEqual([m['id'] for m in result],
                         ['gemini-3.8-flash', 'gemini-4-argon'])
        self.assertTrue(result[0]['recommendation']['free_tier'])
        self.assertFalse(result[1]['recommendation']['free_tier'])
        self.assertEqual(result[0]['input_limit'], 1048576)

    async def test_all_four_supported_ids_in_guide(self):
        for model_id in ('gemini-3.5-flash-lite', 'gemini-3.8-flash',
                         'gemini-3.1-pro-preview', 'gemini-4-argon'):
            info = catalog.describe_gemini_model(model_id)
            self.assertTrue(info['role'])
            self.assertTrue(info['thinking_levels'])
        self.assertNotIn('minimal', catalog.describe_gemini_model('gemini-3.8-flash')['thinking_levels'])
        self.assertNotIn('minimal', catalog.describe_gemini_model('gemini-4-argon')['thinking_levels'])

    async def test_multi_page_gemini_models(self):
        responses = [
            {'models': [{'name': 'models/gemini-3.5-flash-lite',
                         'supportedGenerationMethods': ['generateContent']}],
             'nextPageToken': 'next'},
            {'models': [{'name': 'models/gemini-3.8-flash',
                         'supportedGenerationMethods': ['generateContent']}]},
        ]
        with patch.object(catalog, '_request_json', side_effect=responses) as req:
            result = catalog._gemini_models('fake')
        self.assertEqual(len(result), 2)
        self.assertIn('pageToken=next', req.call_args_list[1].args[0])

    async def test_duplicate_pagination_token_rejected(self):
        with patch.object(catalog, '_request_json', return_value={
            'models': [], 'nextPageToken': 'same'}):
            with self.assertRaises(catalog.ModelCatalogError):
                catalog._gemini_models('fake')

    async def test_openai_caches_for_same_key(self):
        with patch.object(catalog, '_provider_key', return_value='fake'), \
             patch.object(catalog, '_openai_models', return_value=[{'id': 'x'}]) as getter:
            first = await catalog.list_provider_models('openai')
            second = await catalog.list_provider_models('openai')
        self.assertEqual(first['status'], 'live')
        self.assertEqual(second['status'], 'cached')
        getter.assert_called_once()

    async def test_refresh_bypasses_cache(self):
        with patch.object(catalog, '_provider_key', return_value='fake'), \
             patch.object(catalog, '_openai_models', return_value=[]) as getter:
            await catalog.list_provider_models('openai')
            await catalog.list_provider_models('openai', refresh=True)
        self.assertEqual(getter.call_count, 2)

    async def test_errors_are_descriptive_not_fatal(self):
        with patch.object(catalog, '_provider_key', return_value='fake'), \
             patch.object(catalog, '_gemini_models',
                          side_effect=catalog.ModelCatalogError('HTTP 403')):
            result = await catalog.list_provider_models('gemini')
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('fake', str(result))

    async def test_combined_catalog_always_has_recommendations(self):
        with patch.object(catalog, '_provider_key', return_value=None):
            result = await catalog.model_catalog()
        self.assertIn('gemini-4-argon', result['gemini_guide'])
        self.assertEqual(result['gemini']['status'], 'missing_key')


if __name__ == '__main__':
    unittest.main()
