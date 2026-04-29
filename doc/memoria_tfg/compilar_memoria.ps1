$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$miktexBin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
$pdflatex = (Get-Command pdflatex.exe -ErrorAction SilentlyContinue).Source
$bibtex = (Get-Command bibtex.exe -ErrorAction SilentlyContinue).Source

if (-not $pdflatex) {
    $candidate = Join-Path $miktexBin "pdflatex.exe"
    if (Test-Path $candidate) { $pdflatex = $candidate }
}

if (-not $bibtex) {
    $candidate = Join-Path $miktexBin "bibtex.exe"
    if (Test-Path $candidate) { $bibtex = $candidate }
}

if (-not $pdflatex) {
    throw "No se ha encontrado pdflatex. Instala MiKTeX o reinicia la terminal para actualizar el PATH."
}

& $pdflatex -interaction=nonstopmode -file-line-error -shell-escape __memoria.tex

if ($bibtex) {
    & $bibtex __memoria
}

& $pdflatex -interaction=nonstopmode -file-line-error -shell-escape __memoria.tex
& $pdflatex -interaction=nonstopmode -file-line-error -shell-escape __memoria.tex

Invoke-Item (Join-Path $root "__memoria.pdf")
