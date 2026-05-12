# One-line vault env for PowerShell (same effect as manual $env:... lines).
# Usage from this repo root:
#   . .\vault_on.ps1
# Or run as script (still sets session env):
#   .\vault_on.ps1
param([switch]$Quiet)

$keyPath = Join-Path $PSScriptRoot ".offchain_vault_key"
if (-not (Test-Path -LiteralPath $keyPath)) {
    Write-Error "Missing $keyPath — copy .offchain_vault_key.example and set one line secret."
}

$env:BLOCKCHAIN_OFFCHAIN_VAULT_ENABLED = "1"
$env:BLOCKCHAIN_OFFCHAIN_VAULT_KEY = (Get-Content -LiteralPath $keyPath -Raw).Trim()
if (-not $Quiet) {
    Write-Host "Vault ON (key from .offchain_vault_key)."
}
