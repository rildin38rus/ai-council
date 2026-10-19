"""Mock-only tests; never require tokens or a network connection."""
import json
import unittest
from unittest.mock import patch
from scripts import gemini_pr_review as review

REPO = 'owner/ai-council'
PR = {'user': {'login': 'owner'},
      'head': {'repo': {'full_name': REPO}, 'sha': 'abc123'},
      'base': {'repo': {'full_name': REPO}}}


class ReviewerTests(unittest.TestCase):
    def test_authentication_allows_owner_same_repo(self):
        review.authenticate_pr(PR, REPO)

    def test_rejects_fork_and_foreign_author(self):
        for modified in ({'user': {'login': 'contributor'}},
                         {'head': {'repo': {'full_name': 'elsewhere/ai-council'}, 'sha': 'abc123'}}):
            case = json.loads(json.dumps(PR))
            case.update(modified)
            with self.assertRaises(review.ReviewError):
                review.authenticate_pr(case, REPO)

    def test_redact_high_confidence_keys(self):
        source = 'xx AIza' + 'A' * 32 + ' sk-' + 'b' * 25
        safe = review.redact_secrets(source)
        self.assertNotIn('AIza', safe)
        self.assertNotIn('sk-', safe)
        self.assertIn('REDACTED', safe)

    def test_patches_truncate_and_report_omissions(self):
        files = [{'filename': 'app.py', 'patch': '+a\n' * 5000,
                  'status': 'modified', 'additions': 5000, 'deletions': 0},
                 {'filename': '.env', 'patch': '+secret=123', 'status': 'added'},
                 {'filename': 'static/icon.png', 'status': 'added'}]
        with patch.object(review, 'github', return_value=files):
            diff, coverage = review.collect_patches(REPO, 5, 'fake', max_chars=12000)
        self.assertLess(len(diff), 12000)
        self.assertEqual(coverage['included'], 1)
        self.assertEqual(coverage['skipped'], 2)
        self.assertEqual(coverage['truncated'], 1)
        self.assertNotIn('secret=123', diff)

    def test_rejects_no_text_diff(self):
        with patch.object(review, 'github', return_value=[{'filename': 'file.bin'}]):
            with self.assertRaisesRegex(review.ReviewError, 'No reviewable'):
                review.collect_patches(REPO, 8, 'fake')

    def test_pagination_and_complete_coverage(self):
        first = [{'filename': f'file{i}.py', 'patch': '+x'} for i in range(100)]
        with patch.object(review, 'github', side_effect=[first, []]) as mocked:
            diff, coverage = review.collect_patches(REPO, 1, 'fake')
        self.assertEqual(coverage['included'], 30)
        self.assertEqual(coverage['skipped'], 70)
        self.assertEqual(mocked.call_count, 2)
        self.assertIn('FILE 30', diff)

    def test_reuse_prior_bot_comment_skips_paid_call(self):
        comments = [{'body': review.MARKER + '\n`abc123`', 'user': {'login': 'github-actions[bot]'}}]
        with patch.object(review, 'github', side_effect=[PR, comments]) as g, \
             patch.object(review, 'gemini_review') as gemini:
            result = review.review_pr(REPO, 1, 'fake', 'fake', 'gemini-2.5-flash')
        self.assertIn('Already reviewed', result)
        self.assertEqual(g.call_count, 2)
        gemini.assert_not_called()

    def test_gemini_api_payload_and_text(self):
        response = {'candidates': [{'content': {'parts': [{'text': 'P1: Fix API'}, {'text': 'P2: Add test'}]},
                                    'finishReason': 'STOP'}]}
        with patch.object(review, 'api_json', return_value=response) as api:
            text = review.gemini_review('+changed', {'included': 1}, 'fake', 'gemini-2.5-flash', 1024)
        self.assertIn('P1: Fix', text)
        args, kwargs = api.call_args
        self.assertEqual(kwargs['payload']['generationConfig']['thinkingConfig']['thinkingBudget'], 1024)
        self.assertEqual(kwargs['api_key'], 'fake')
        self.assertIn('UNTRUSTED_DIFF', kwargs['payload']['contents'][0]['parts'][0]['text'])

    def test_empty_model_answer_is_error(self):
        with patch.object(review, 'api_json', return_value={'candidates': []}):
            with self.assertRaisesRegex(review.ReviewError, 'no candidates'):
                review.gemini_review('diff', {}, 'fake', 'gemini-2.5-flash')

    def test_post_review_only_after_success(self):
        calls = []
        def fake_github(repo, path, token, method='GET', payload=None):
            calls.append((path, method, payload))
            if path == 'pulls/7':
                return PR
            if 'comments?' in path:
                return []
            if 'files?' in path:
                return [{'filename': 'app.py', 'patch': '@@ -1 +1 @@\n+changed',
                         'additions': 1, 'deletions': 0}]
            if path == 'issues/7/comments' and method == 'POST':
                return {'id': 9}
            raise AssertionError(path)
        with patch.object(review, 'github', side_effect=fake_github), \
             patch.object(review, 'gemini_review', return_value='P1: suspicious race'):
            result = review.review_pr(REPO, 7, 'fake', 'fake', 'gemini-2.5-flash')
        self.assertIn('Published', result)
        self.assertEqual(calls[-1][1], 'POST')
        self.assertIn(review.MARKER, calls[-1][2]['body'])
        self.assertIn('abc123', calls[-1][2]['body'])

    def test_no_comment_after_gemini_failure(self):
        def fake_github(repo, path, token, method='GET', payload=None):
            if method == 'POST':
                raise AssertionError('Should not publish an error as reviewed')
            if path.startswith('pulls/1/files'):
                return [{'filename': 'x.py', 'patch': '+x'}]
            if path.startswith('issues/1/comments'):
                return []
            return PR
        with patch.object(review, 'github', side_effect=fake_github), \
             patch.object(review, 'gemini_review', side_effect=review.ReviewError('API failed')):
            with self.assertRaises(review.ReviewError):
                review.review_pr(REPO, 1, 'fake', 'fake', 'gemini-2.5-flash')


if __name__ == '__main__':
    unittest.main()
