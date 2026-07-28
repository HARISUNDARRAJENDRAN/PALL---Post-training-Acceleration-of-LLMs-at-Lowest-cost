[CmdletBinding()]
param(
    [string]$WorkRoot = "C:\PALL_DATA"
)

# Audit revision 2: tolerate endpoints that reject HEAD and retrigger the controlled workflow.
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

    foreach ($method in @("Head", "Get")) {
        try {
            $params = @{
                Uri = $Uri
                Method = $method
                TimeoutSec = 20
                UseBasicParsing = $true
            }
            if ($method -eq "Get") {
                $params["Headers"] = @{ Range = "bytes=0-0" }
            }
            $response = Invoke-WebRequest @params
            return [ordered]@{
                reachable = $true
                status = [int]$response.StatusCode
                method = $method
                error = $null
            }
        }
        catch {
            $lastError = $_
        }
    }

    $status = $null
    if ($lastError.Exception.Response -and $lastError.Exception.Response.StatusCode) {
        $status = [int]$lastError.Exception.Response.StatusCode
    }
    return [ordered]@{
        reachable = $false
        status = $status
        method = $null
        error = $lastError.Exception.Message
    }
}

function Get-EnvPresence {
    param([Parameter(Mandatory = $true)][string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    return -not [string]::IsNullOrWhiteSpace($value)
}

function Get-SafeEnv {
    param([Parameter(Mandatory = $true)][string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($null -eq $value) { return "" }
    return $value
}

$githubRepository = Get-SafeEnv -Name "GITHUB_REPOSITORY"
$githubRef = Get-SafeEnv -Name "GITHUB_REF"
$githubSha = Get-SafeEnv -Name "GITHUB_SHA"
$githubActor = Get-SafeEnv -Name "GITHUB_ACTOR"
$githubEventName = Get-SafeEnv -Name "GITHUB_EVENT_NAME"
$runnerName = Get-SafeEnv -Name "RUNNER_NAME"
$runnerOS = Get-SafeEnv -Name "RUNNER_OS"
$runnerArch = Get-SafeEnv -Name "RUNNER_ARCH"
$workspace = Get-SafeEnv -Name "GITHUB_WORKSPACE"
if ([string]::IsNullOrWhiteSpace($workspace)) {
    $workspace = (Get-Location).Path
}

Write-Host "=== PALL self-hosted runner audit ==="
Write-Host "Repository: $githubRepository"
Write-Host "Ref:        $githubRef"
Write-Host "Runner:     $runnerName"
Write-Host "Workspace:  $workspace"

New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
$reportDir = Join-Path $workspace "reports\runner-audit"
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
    audit_revision = 2
    github = [ordered]@{
        repository = $githubRepository
        ref = $githubRef
        sha = $githubSha
        actor = $githubActor
        event_name = $githubEventName
        runner_name = $runnerName
        runner_os = $runnerOS
        runner_arch = $runnerArch
        workspace = $workspace
    }
    system = [ordered]@{
        computer_name = Get-SafeEnv -Name "COMPUTERNAME"
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
