Get-ChildItem *.py | ForEach-Object { autoflake --in-place --remove-all-unused-imports $_.FullName }
Get-ChildItem **/*.py | ForEach-Object { autoflake --in-place --remove-all-unused-imports $_.FullName }
black .
