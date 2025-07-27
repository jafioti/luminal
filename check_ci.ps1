Write-Host "Running Local CI Checks..." -ForegroundColor Green
Write-Host ""

# Fmt Check
Write-Host "1. Running Fmt Check..." -ForegroundColor Yellow
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Fmt check failed! Run 'cargo fmt --all' to fix." -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Fmt check passed!" -ForegroundColor Green
}
Write-Host ""

# Clippy Check
Write-Host "2. Running Clippy..." -ForegroundColor Yellow
cargo clippy --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Clippy failed!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Clippy passed!" -ForegroundColor Green
}
Write-Host ""

# Unit Tests
Write-Host "3. Running Unit Tests..." -ForegroundColor Yellow
cargo test --workspace --verbose
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Unit tests failed!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Unit tests passed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 All CI checks passed! Safe to push." -ForegroundColor Green 