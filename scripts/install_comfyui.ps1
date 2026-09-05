[CmdletBinding()]
param(
    [string]$AiRoot = 'C:\AI'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$installMutex = [Threading.Mutex]::new(
    $false,
    'Local\DoomsdaySupplyBureau-ComfyUI-Install'
)
$ownsInstallMutex = $false
try {
    $ownsInstallMutex = $installMutex.WaitOne(0)
} catch [Threading.AbandonedMutexException] {
    $ownsInstallMutex = $true
}
if (-not $ownsInstallMutex) {
    $installMutex.Dispose()
    throw 'Another Doomsday Supply Bureau ComfyUI installer is already running.'
}

try {
$downloadRoot = Join-Path $AiRoot 'Downloads'
$installRoot = Join-Path $AiRoot 'ComfyUI_windows_portable'
$archivePath = Join-Path $downloadRoot 'ComfyUI_windows_portable_nvidia_v0.34.0.7z'
$installMarker = Join-Path $installRoot '.dsb-v0.34.0-complete'
$aria2Command = Get-Command aria2c -ErrorAction SilentlyContinue |
    Select-Object -First 1
$aria2Path = if ($aria2Command) { $aria2Command.Source } else { $null }
if (-not $aria2Path) {
    $wingetPackages = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $aria2Path = Get-ChildItem -LiteralPath $wingetPackages -Filter aria2c.exe `
        -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

$assets = @(
    [pscustomobject]@{
        Name = 'ComfyUI v0.34.0 NVIDIA Portable'
        Url = 'https://github.com/Comfy-Org/ComfyUI/releases/download/v0.34.0/ComfyUI_windows_portable_nvidia.7z'
        Path = $archivePath
        Bytes = 2146721943L
        Sha256 = 'ed57cc6b19ae3d83add1ecebfdd56b25e04e0008cf0fe9af43a4ad8797e2a24c'
    },
    [pscustomobject]@{
        Name = 'FLUX.2 Klein 4B distilled FP8'
        Url = 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/5b4408e59397a4a37ccb46afe426d8ed86379441/flux-2-klein-4b-fp8.safetensors?download=true'
        Path = Join-Path $downloadRoot 'flux-2-klein-4b-fp8.safetensors'
        Bytes = 4070624520L
        Sha256 = '97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6'
    },
    [pscustomobject]@{
        Name = 'Qwen 3 4B text encoder'
        Url = 'https://huggingface.co/Comfy-Org/z_image_turbo/resolve/08d04455279082882deaabc8d0d09fc914c071e1/split_files/text_encoders/qwen_3_4b.safetensors?download=true'
        Path = Join-Path $downloadRoot 'qwen_3_4b.safetensors'
        Bytes = 8044982048L
        Sha256 = '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a'
    },
    [pscustomobject]@{
        Name = 'FLUX.2 VAE'
        Url = 'https://huggingface.co/Comfy-Org/flux2-dev/resolve/ab9055628ea245000e610f2aa2c96f4746093546/split_files/vae/flux2-vae.safetensors?download=true'
        Path = Join-Path $downloadRoot 'flux2-vae.safetensors'
        Bytes = 336213556L
        Sha256 = 'd64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5'
    }
)

function Test-Asset {
    param([Parameter(Mandatory)]$Asset)

    if (-not (Test-Path -LiteralPath $Asset.Path -PathType Leaf)) {
        return $false
    }
    $file = Get-Item -LiteralPath $Asset.Path
    if ($file.Length -ne $Asset.Bytes) {
        return $false
    }
    $actualHash = (Get-FileHash -LiteralPath $Asset.Path -Algorithm SHA256).Hash
    return $actualHash.Equals($Asset.Sha256, [StringComparison]::OrdinalIgnoreCase)
}

function Get-Asset {
    param([Parameter(Mandatory)]$Asset)

    $ariaPartial = Test-Path -LiteralPath "$($Asset.Path).aria2" -PathType Leaf
    if ($ariaPartial) {
        $leafName = Split-Path -Leaf $Asset.Path
        $activeAria = Get-CimInstance Win32_Process -Filter "Name = 'aria2c.exe'" `
            -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -and
                $_.CommandLine.IndexOf(
                    $leafName,
                    [StringComparison]::OrdinalIgnoreCase
                ) -ge 0
            }
        if ($activeAria) {
            throw "An aria2 process is already downloading $leafName. Wait for it to finish."
        }
    }
    if (-not $ariaPartial -and (Test-Asset -Asset $Asset)) {
        Write-Host "[OK] $($Asset.Name)"
        return
    }

    if ($ariaPartial -and -not $aria2Path) {
        throw "An aria2 partial download exists, but aria2c was not found: $($Asset.Path)"
    }

    if (-not $ariaPartial -and
        (Test-Path -LiteralPath $Asset.Path -PathType Leaf) -and
        (Get-Item -LiteralPath $Asset.Path).Length -ge $Asset.Bytes) {
        $backup = "$($Asset.Path).invalid-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Move-Item -LiteralPath $Asset.Path -Destination $backup
        Write-Warning "Preserved an invalid complete file as $backup"
    }

    Write-Host "[DOWNLOAD] $($Asset.Name)"
    if ($aria2Path) {
        & $aria2Path `
            --continue=true `
            --max-connection-per-server=16 `
            --split=16 `
            --min-split-size=1M `
            --file-allocation=none `
            --auto-file-renaming=false `
            --allow-overwrite=true `
            --console-log-level=warn `
            --summary-interval=10 `
            "--dir=$(Split-Path -Parent $Asset.Path)" `
            "--out=$(Split-Path -Leaf $Asset.Path)" `
            $Asset.Url
    } else {
        & curl.exe --location --fail --retry 5 --retry-delay 3 --continue-at - `
            --output $Asset.Path $Asset.Url
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($Asset.Name)"
    }
    if (-not (Test-Asset -Asset $Asset)) {
        throw "Size or SHA256 verification failed: $($Asset.Name)"
    }
    Write-Host "[VERIFIED] $($Asset.Name)"
}

New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
foreach ($asset in $assets) {
    Get-Asset -Asset $asset
}

$requiredPortablePaths = @(
    (Join-Path $installRoot 'ComfyUI\main.py'),
    (Join-Path $installRoot 'python_embeded\python.exe'),
    (Join-Path $installRoot 'run_nvidia_gpu.bat')
)
$portableFilesReady = @(
    $requiredPortablePaths |
        Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
).Count -eq 0

if (-not (Test-Path -LiteralPath $installMarker -PathType Leaf) -or
    -not $portableFilesReady) {
    $sevenZip = 'C:\Program Files\7-Zip\7z.exe'
    if (-not (Test-Path -LiteralPath $sevenZip -PathType Leaf)) {
        throw '7-Zip was not found at C:\Program Files\7-Zip\7z.exe.'
    }
    Write-Host '[EXTRACT] ComfyUI portable'
    & $sevenZip x $archivePath "-o$AiRoot" -y
    if ($LASTEXITCODE -ne 0) {
        throw 'ComfyUI extraction failed.'
    }
}

$missingPortablePaths = @(
    $requiredPortablePaths |
        Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
)
if ($missingPortablePaths.Count -gt 0) {
    throw "ComfyUI extraction is incomplete; missing: $($missingPortablePaths -join ', ')"
}
New-Item -ItemType File -Path $installMarker -Force | Out-Null

$modelCopies = @(
    @(
        (Join-Path $downloadRoot 'flux-2-klein-4b-fp8.safetensors'),
        (Join-Path $installRoot 'ComfyUI\models\diffusion_models\flux-2-klein-4b-fp8.safetensors'),
        '97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6'
    ),
    @(
        (Join-Path $downloadRoot 'qwen_3_4b.safetensors'),
        (Join-Path $installRoot 'ComfyUI\models\text_encoders\qwen_3_4b.safetensors'),
        '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a'
    ),
    @(
        (Join-Path $downloadRoot 'flux2-vae.safetensors'),
        (Join-Path $installRoot 'ComfyUI\models\vae\flux2-vae.safetensors'),
        'd64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5'
    )
)

foreach ($copy in $modelCopies) {
    $source = $copy[0]
    $destination = $copy[1]
    $expectedHash = $copy[2]
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force |
        Out-Null

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $currentHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($currentHash.Equals(
                $expectedHash,
                [StringComparison]::OrdinalIgnoreCase
            )) {
            Write-Host "[OK] $(Split-Path -Leaf $destination) installed"
            continue
        }
        $backup = "$destination.invalid-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Move-Item -LiteralPath $destination -Destination $backup
        Write-Warning "Preserved an invalid installed model as $backup"
    }

    $temporary = "$destination.installing"
    if (Test-Path -LiteralPath $temporary -PathType Leaf) {
        $backup = "$temporary.invalid-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Move-Item -LiteralPath $temporary -Destination $backup
    }
    Write-Host "[COPY] $(Split-Path -Leaf $destination)"
    Copy-Item -LiteralPath $source -Destination $temporary
    $installedHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
    if (-not $installedHash.Equals(
            $expectedHash,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "Copied model SHA256 mismatch: $temporary"
    }
    Move-Item -LiteralPath $temporary -Destination $destination
}

Write-Host "[READY] ComfyUI installed at $installRoot"
} finally {
    if ($ownsInstallMutex) {
        $installMutex.ReleaseMutex()
    }
    $installMutex.Dispose()
}
