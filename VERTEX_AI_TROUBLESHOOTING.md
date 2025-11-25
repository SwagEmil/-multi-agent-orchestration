# Troubleshooting Vertex AI Model Access

If you see models in the "Model Garden" but get **404 Not Found** errors when trying to use them, follow these steps to unlock access.

---

## 1. Check Billing Status (Most Common Cause)

Even if you have a billing account, **Vertex AI requires an active billing account with good standing**.

1. Go to **Billing**: https://console.cloud.google.com/billing
2. Select your project: `code-review-479122`
3. Verify:
   - Is a billing account linked?
   - Are there any payment issues?
   - Is it a "Free Trial" account? (Free trials have strict model limitations)

**Fix**: Link a valid credit card or upgrade from Free Trial to "Blaze" (Pay-as-you-go). You still get free tier usage, but it unlocks the API.

---

## 2. Check Quotas

New projects often start with **0 quota** for powerful models like Gemini 1.5 Pro.

1. Go to **IAM & Admin** > **Quotas**: https://console.cloud.google.com/iam-admin/quotas?project=code-review-479122
2. Filter for: `Vertex AI API`
3. Search for: `base_model_gemini_1.5_pro` (or similar)
4. Check the **Limit** column.
   - If it says `0`, you cannot use the model.

**Fix**:
1. Select the quota.
2. Click **EDIT QUOTAS**.
3. Request an increase (e.g., to 60 requests/min).
4. Google usually approves this instantly or within 24 hours.

---

## 3. Check Region Availability

Not all models are available in `us-central1`.

1. Go to **Vertex AI** > **Model Garden**.
2. Click on **Gemini 1.5 Pro**.
3. Look for "Region availability".

**Fix**:
If the model is only available in `us-west1` or `us-east4`:
1. Open `.env` file.
2. Change `GOOGLE_CLOUD_LOCATION` to the supported region.
   ```bash
   GOOGLE_CLOUD_LOCATION=us-west1
   ```

---

## 4. "Generative AI" vs "Vertex AI"

Sometimes you need to enable the specific **Generative AI API** separately.

1. Go to **APIs & Services** > **Library**.
2. Search for **"Generative AI API"**.
3. Click **ENABLE** if not already enabled.

---

## Summary Checklist

- [ ] Billing Account Linked & Active?
- [ ] Quota > 0 for Gemini 1.5 Pro?
- [ ] Correct Region selected in `.env`?
- [ ] Generative AI API Enabled?

If you check all these and it still fails, stick with **Gemini 2.0 Flash Experimental** (`gemini-2.0-flash-exp`) which is currently working perfectly for your project.
