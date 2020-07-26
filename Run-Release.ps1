.\Build-Release.ps1

$env:RUST_LOG="info"
.\target\release\mnist_classification.exe
