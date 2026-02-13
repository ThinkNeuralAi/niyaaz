$response = Invoke-WebRequest -Uri 'http://localhost:5000/api/get_active_channels?store_id=store_1' -ErrorAction Ignore
if ($response) {
    $data = $response.Content | ConvertFrom-Json
    Write-Host "✅ API Response received"
    Write-Host "Channels: $($data.count)"
    if ($data.active_channels) {
        Write-Host "`nFirst 3 channels:"
        $data.active_channels | Select-Object -First 3 | ForEach-Object {
            Write-Host "  - $($_.channel_id): $($_.modules | Convert-ToJson -Compress)"
        }
    }
} else {
    Write-Host "❌ API not responding"
}
