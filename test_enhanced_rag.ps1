# Test Enhanced Self-RAG with Advanced Features
# PowerShell Script

$BASE = "https://ahmedaayman-enterprise-ai-document-intelligence-9f1cbfc.hf.space"

Write-Host "=== Testing Enhanced Self-RAG (Multi-Doc + Citations + Query Rewriting) ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: Single Document with Citations
Write-Host "Test 1: Single Document with Citation Tracking" -ForegroundColor Yellow
$body = @{
    context_text = "The Q3 2024 revenue was $5.2 million, up 25% from Q2. Customer satisfaction reached 4.8/5.0."
    question = "What was the revenue?"
    doc_name = "Q3 Financial Report"
    use_query_rewriting = $true
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE/chat/" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Answer: $($response.answer)" -ForegroundColor Green
    Write-Host "Confidence: $($response.confidence)" -ForegroundColor Cyan
    Write-Host "Citations:" -ForegroundColor Yellow
    foreach ($cite in $response.citations) {
        Write-Host "  - $($cite.doc_name) (score: $($cite.similarity_score))" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n---`n"

# Test 2: Multi-Document Query
Write-Host "Test 2: Multi-Document RAG" -ForegroundColor Yellow
$body = @{
    documents = @(
        @{
            text = "Apple Inc. Q1 2024: Revenue $119.6 billion. iPhone sales dominated."
            name = "Apple Q1 Report"
        },
        @{
            text = "Microsoft Q1 2024: Revenue $62 billion. Cloud growth at 30%."
            name = "Microsoft Q1 Report"
        },
        @{
            text = "Tesla Q1 2024: Revenue $21.3 billion. Model Y best seller."
            name = "Tesla Q1 Report"
        }
    )
    question = "Which company had the highest revenue in Q1 2024?"
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$BASE/chat/multi-doc" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Question: $($response.question)" -ForegroundColor Cyan
    Write-Host "Answer: $($response.answer)" -ForegroundColor Green
    Write-Host "Confidence: $($response.confidence)" -ForegroundColor Cyan
    Write-Host "Documents Searched: $($response.documents_searched)" -ForegroundColor Gray
    Write-Host "Relevant Chunks: $($response.relevant_chunks)/$($response.retrieved_chunks)" -ForegroundColor Gray
    Write-Host "`nCitations:" -ForegroundColor Yellow
    foreach ($cite in $response.citations) {
        Write-Host "  - $($cite.doc_name)" -ForegroundColor Gray
        Write-Host "    Score: $($cite.similarity_score), Text: $($cite.chunk_text.Substring(0, [Math]::Min(60, $cite.chunk_text.Length)))..." -ForegroundColor DarkGray
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n---`n"

# Test 3: Query Rewriting
Write-Host "Test 3: Query Rewriting Feature" -ForegroundColor Yellow
$body = @{
    context_text = "The new product launch in September 2024 was successful. Sales exceeded projections by 40%."
    question = "How did the new thing do?"
    use_query_rewriting = $true
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE/chat/" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Original Question: $($response.question)" -ForegroundColor Cyan
    if ($response.rewritten_query) {
        Write-Host "Rewritten Query: $($response.rewritten_query)" -ForegroundColor Yellow
    }
    Write-Host "Answer: $($response.answer)" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n---`n"

# Test 4: List Indexed Documents
Write-Host "Test 4: List Indexed Documents" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE/chat/documents"
    Write-Host "Indexed Documents: $($response.documents.Count)" -ForegroundColor Cyan
    foreach ($doc in $response.documents) {
        Write-Host "  - $($doc.name) (ID: $($doc.doc_id), Chunks: $($doc.chunk_count))" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n---`n"

# Test 5: Enhanced Health Check
Write-Host "Test 5: Enhanced Health Check" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE/chat/health"
    Write-Host "Status: $($response.status)" -ForegroundColor Green
    Write-Host "Model: $($response.model)" -ForegroundColor Cyan
    Write-Host "Workflow: $($response.workflow)" -ForegroundColor Cyan
    Write-Host "Features:" -ForegroundColor Yellow
    foreach ($feature in $response.features) {
        Write-Host "  ✓ $feature" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== Enhanced Self-RAG Tests Complete ===" -ForegroundColor Cyan
