Executar no Windows

Estrutura esperada:
- `treinamento\datasets\original`
- `treinamento\datasets\augmented`
- `treinamento\outputs`
- scripts `.py` e notebooks na raiz de `treinamento`

Passos:
1. Abra o PowerShell em `C:\Users\ghmd1\Desktop\treinamento`
2. Rode:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_windows_cpu.ps1
```

O script:
- cria a venv `.venv_tf_win` se ela nao existir
- instala as dependencias de `requirements_tf.txt`
- executa `run_all_experiments_supervisor.py` em modo CPU
- reaproveita checkpoints salvos em `outputs`

Comandos uteis:

Ver log:

```powershell
Get-Content .\run_all_experiments_child_windows.log -Wait
```

Ver supervisor:

```powershell
Get-Content .\run_all_experiments_supervisor_windows.log -Wait
```

Parar:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_all_experiments_supervisor.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```
