# Quick Fix: Chatbot & Knowledge Graph Not Working

## ✅ Status Check

**Backend:** ✅ All endpoints working
- Health: ✅ OK
- Chat Sessions: ✅ OK  
- Documents: ✅ OK (2 documents uploaded)
- Knowledge Graph: ✅ OK (4 entities, 1 relationship)

**Frontend:** ⚠️ Needs browser refresh

---

## 🔧 Fix Steps

### Step 1: Hard Refresh Browser
```
Press: Ctrl + Shift + R (or Ctrl + F5)
```
This clears the browser cache and loads the latest frontend code.

### Step 2: Clear Browser Cache (if step 1 doesn't work)
**Chrome/Edge:**
1. Press `F12` to open DevTools
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"

**Firefox:**
1. Press `Ctrl + Shift + Delete`
2. Select "Cached Web Content"
3. Click "Clear Now"
4. Refresh page

### Step 3: Restart Frontend Container
```bash
docker-compose restart frontend
```

Wait 10 seconds, then refresh browser.

### Step 4: Check Browser Console
1. Open DevTools (`F12`)
2. Go to "Console" tab
3. Look for errors (red text)
4. Share any errors you see

---

## 🧪 Test Chatbot

1. Go to http://localhost:3000/dashboard
2. You should see your 2 uploaded documents
3. Select one or both documents using the selector bar
4. Type a message like "Summarize this document"
5. Press Enter
6. Bot should respond with personality!

**Expected Response Style:**
```
"Oh, great question! 🎉 Let me break down what I found..."
```

---

## 🌐 Test Knowledge Graph

1. Click "Knowledge Graph" in sidebar
2. You should see:
   - 4 nodes (2 document type nodes + 2 document nodes)
   - 1 edge connecting them
3. If blank, check browser console for errors

---

## 🐛 If Still Not Working

### Check Frontend Logs
```bash
docker-compose logs -f frontend
```

Look for:
- API connection errors
- CORS errors
- JavaScript errors

### Check Backend Logs
```bash
docker-compose logs -f backend
```

Look for:
- Request errors
- Database errors
- LLM API errors

### Verify API Connection
Open browser console (`F12`) and run:
```javascript
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(d => console.log(d))
```

Should output:
```json
{status: "healthy", service: "DocuCentric", ...}
```

---

## 📝 Common Issues

### Issue 1: "Failed to fetch"
**Cause:** Backend not running or wrong URL

**Fix:**
```bash
docker-compose ps
# Should show backend as "Up" and "healthy"

# If not:
docker-compose restart backend
```

### Issue 2: Chat sends but no response
**Cause:** Groq API key not configured

**Fix:**
```bash
# Check if API key is set
docker-compose exec backend env | findstr GROQ_API_KEY

# If empty or "your-key-here", update backend/.env:
GROQ_API_KEY=your-actual-key

# Restart backend
docker-compose restart backend
```

### Issue 3: Knowledge graph shows "No data"
**Cause:** No documents uploaded or processed

**Fix:**
1. Upload at least 1 document
2. Wait for processing to complete
3. Refresh knowledge graph page

### Issue 4: Duplicate key warning
**Status:** ✅ FIXED

The duplicate key issue in nav-history has been resolved.

---

## 🎯 Quick Verification Checklist

- [ ] Backend healthy: `curl http://localhost:8000/health`
- [ ] Frontend running: http://localhost:3000
- [ ] Documents uploaded: Check sidebar
- [ ] Browser refreshed: `Ctrl + Shift + R`
- [ ] Console clear: No red errors in F12

---

## 📞 Still Having Issues?

Run the diagnostic:
```bash
diagnose.cmd
```

Share the output along with:
1. Browser console errors (F12 → Console)
2. Frontend logs: `docker-compose logs frontend`
3. Backend logs: `docker-compose logs backend`
