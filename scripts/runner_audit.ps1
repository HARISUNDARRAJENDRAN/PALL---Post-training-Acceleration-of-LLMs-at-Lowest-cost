[CmdletBinding()]
param(
    [string]$WorkRoot = "C:\PALL_DATA"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-CommandVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Arguments = @("--version")
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        return $null
    }

    try {
        $output = & $cmd.Source @Arguments 2>&1 | Out-String
        return $output.Trim()
    }
    catch {
        return "available, but version check failed: $($_.Exception.Message)"
    }
}

function Test-Endpoint {
    param([Parameter(Mandatory = $true)][string]$Uri)

    try {
        $response = Invoke-WebRequest -Uri $Uri -Method Head -TimeoutSec 20 -UseBasicParsing
        return [ordered]@{
            reachable = $true
            status = [int]$response.StatusCode
            error = $null
        }
    }
    catch {
        $status = $null
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $status = [int]$_.Exception.Response.StatusCode
        }
        return [ordered]@{
            reachable = $false
            status = $status
            error = $_.Exception.Message
        }
    }
}

function Get-EnvPresence {
    param([Parameter(Mandatory = $true)][string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    return -not [string]::IsNullOrWhiteSpace($value)
}

Write-Host "=== PALL self-hosted runner audit ==="
Write-Host "Repository: $env:GITHUB_REPOSITORY"
Write-Host "Ref:        $env:GITHUB_REF"
Write-Host "Runner:     $env:RUNNER_NAME"
Write-Host "Workspace:  $env:GITHUB_WORKSPACE"

New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
$reportDir = Join-Path $env:GITHUB_WORKSPACE "reports\runner-audit"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

$os = Get-CimInstance Win32_OperatingSystem
$computer = Get-CimInstance Win32_ComputerSystem
$disks = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    [ordered]@{
        device = $_.DeviceID
        size_gb = [math]::Round($_.Size / 1GB, 2)
        free_gb = [math]::Round($_.FreeSpace / 1GB, 2)
        free_percent = if ($_.Size) { [math]::Round(100 * $_.FreeSpace / $_.Size, 2) } else { $null }
    }
}

$gpuOutput = $null
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    try {
        $gpuOutput = (& $nvidia.Source --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | Out-String).Trim()
    }
    catch {
        $gpuOutput = "nvidia-smi failed: $($_.Exception.Message)"
    }
}

$gitStatus = $null
try {
    $gitStatus = (& git status --short --branch 2>&1 | Out-String).Trim()
}
catch {
    $gitStatus = "git status failed: $($_.Exception.Message)"
}

$report = [ordered]@{
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    github = [ordered]@{
        repository = $env:GITHUB_REPOSITORY
        ref = $env:GITHUB_REF
        sha = $env:GITHUB_SHA
        actor = $env:GITHUB_ACTOR
        event_name = $env:GITHUB_EVENT_NAME
        runner_name = $env:RUNNER_NAME
        runner_os = $env:RUNNER_OS
        runner_arch = $env:RUNNER_ARCH
        workspace = $env:GITHUB_WORKSPACE
    }
    system = [ordered]@{
        computer_name = $env:COMPUTERNAME
        os_caption = $os.Caption
        os_version = $os.Version
        os_build = $os.BuildNumber
        total_memory_gb = [math]::Round($computer.TotalPhysicalMemory / 1GB, 2)
        powershell_version = $PSVersionTable.PSVersion.ToString()
        processor_count = [Environment]::ProcessorCount
        disks = @($disks)
        gpu = $gpuOutput
    }
    tools = [ordered]@{
        git = Get-CommandVersion -Name "git"
        python = Get-CommandVersion -Name "python"
        py_launcher = Get-CommandVersion -Name "py" -Arguments @("--version")
        pip = Get-CommandVersion -Name "pip"
        huggingface_cli = Get-CommandVersion -Name "hf" -Arguments @("--help")
        nvidia_smi = if ($nvidia) { $nvidia.Source } else { $null }
    }
    credentials_present = [ordered]@{
        HF_TOKEN_READ = Get-EnvPresence -Name "HF_TOKEN_READ"
        HF_TOKEN = Get-EnvPresence -Name "HF_TOKEN"
        PALL_LLM_API_KEY = Get-EnvPresence -Name "PALL_LLM_API_KEY"
        PALL_LLM_BASE_URL = Get-EnvPresence -Name "PALL_LLM_BASE_URL"
        PALL_LLM_MODEL = Get-EnvPresence -Name "PALL_LLM_MODEL"
    }
    network = [ordered]@{
        github = Test-Endpoint -Uri "https://github.com"
        huggingface = Test-Endpoint -Uri "https://huggingface.co"
        huggingface_dataset = Test-Endpoint -Uri "https://huggingface.co/datasets/Harisundar/pall"
    }
    paths = [ordered]@{
        persistent_work_root = $WorkRoot
        persistent_work_root_exists = Test-Path $WorkRoot
    }
    git_status = $gitStatus
}

$reportPath = Join-Path $reportDir "runner_audit.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $reportPath -Encoding UTF8

Write-Host ""
Write-Host "=== Audit summary ==="
Write-Host "OS:          $($report.system.os_caption) $($report.system.os_version)"
Write-Host "RAM:         $($report.system.total_memory_gb) GB"
Write-Host "Python:      $($report.tools.python)"
Write-Host "Git:         $($report.tools.git)"
Write-Host "GPU:         $($report.system.gpu)"
Write-Host "HF network:  $($report.network.huggingface.reachable)"
Write-Host "HF read env: $($report.credentials_present.HF_TOKEN_READ -or $report.credentials_present.HF_TOKEN)"
Write-Host "LLM env:     $($report.credentials_present.PALL_LLM_API_KEY -and $report.credentials_present.PALL_LLM_BASE_URL -and $report.credentials_present.PALL_LLM_MODEL)"
Write-Host "Persistent:  $WorkRoot"
Write-Host "Report:      $reportPath"

if (-not $report.network.github.reachable) {
    throw "GitHub is not reachable from the runner."
}
if (-not $report.network.huggingface.reachable) {
    throw "Hugging Face is not reachable from the runner."
}
if (-not $report.tools.git) {
    throw "Git is not available on the runner."
}
if (-not ($report.tools.python -or $report.tools.py_launcher)) {
    throw "Python is not available on the runner."
}
