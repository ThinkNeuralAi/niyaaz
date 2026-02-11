#!/usr/bin/env pwsh
# Test script to verify all 27 cameras are loaded in the dashboard

Write-Host "`n=========================================="
Write-Host "Testing Dashboard Camera Loading"
Write-Host "=========================================="

# Give the app a moment to start channel loading
Write-Host "`nWaiting for app to load channels..."
Start-Sleep -Seconds 6

# Test Store 1 (should have 15 cameras)
Write-Host "`n📊 Testing Store 1 (should have 15 cameras)..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/get_active_channels?store_id=store_1" -UseBasicParsing -TimeoutSec 10
    $json = $response.Content | ConvertFrom-Json
    $count = $json.count
    $channels = $json.active_channels.channel_id
    
    if ($count -eq 15) {
        Write-Host "✅ SUCCESS! STORE 1 has all 15 cameras:" -ForegroundColor Green
        $channels | Sort-Object {[int]($_ -replace "camera_")} | ForEach-Object {
            Write-Host "   ✓ $_" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ STORE 1 has $count cameras (expected 15)" -ForegroundColor Red
        $channels | Sort-Object {[int]($_ -replace "camera_")} | ForEach-Object {
            Write-Host "   • $_"
        }
    }
} catch {
    Write-Host "❌ ERROR: Could not reach API - $_" -ForegroundColor Red
}

# Test Store 2 (should have 12 cameras)
Write-Host "`n📊 Testing Store 2 (should have 12 cameras)..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/get_active_channels?store_id=store_2" -UseBasicParsing -TimeoutSec 10
    $json = $response.Content | ConvertFrom-Json
    $count = $json.count
    $channels = $json.active_channels.channel_id
    
    if ($count -eq 12) {
        Write-Host "✅ SUCCESS! STORE 2 has all 12 cameras:" -ForegroundColor Green
        $channels | Sort-Object {[int]($_ -replace "camera_")} | ForEach-Object {
            Write-Host "   ✓ $_" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ STORE 2 has $count cameras (expected 12)" -ForegroundColor Red
        $channels | Sort-Object {[int]($_ -replace "camera_")} | ForEach-Object {
            Write-Host "   • $_"
        }
    }
} catch {
    Write-Host "❌ ERROR: Could not reach API - $_" -ForegroundColor Red
}

# Test total
Write-Host "`n📊 Testing Total (should have 27 cameras)..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/get_active_channels" -UseBasicParsing -TimeoutSec 10
    $json = $response.Content | ConvertFrom-Json
    $count = $json.count
    
    if ($count -eq 27) {
        Write-Host "✅ SUCCESS! Dashboard has all 27 cameras!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Dashboard has $count cameras (expected 27)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ ERROR: Could not reach API - $_" -ForegroundColor Red
}

Write-Host "`n=========================================="
Write-Host "Test Complete"
Write-Host "=========================================="
