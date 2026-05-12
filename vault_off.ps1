# Clear vault-related env vars for this PowerShell session.
# Usage: .\vault_off.ps1
Remove-Item Env:BLOCKCHAIN_OFFCHAIN_VAULT_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:BLOCKCHAIN_OFFCHAIN_VAULT_KEY -ErrorAction SilentlyContinue
Write-Host "Vault env cleared (Python will not decrypt vault blobs unless key is set again)."
