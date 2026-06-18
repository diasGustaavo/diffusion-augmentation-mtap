# Guia de Execução — Treinamento EfficientNetV2B0 no WSL GPU

Este guia documenta a arquitetura final e todas as correções aplicadas para que o treino rode continuamente, com GPU, e não seja morto pelo WSL.

---

## 1. Sumário dos problemas encontrados

Durante a execução do `rerun5` (4 experimentos × 7 variantes × 5 folds, dataset generative_plus_traditional), aconteceram três classes distintas de falha:

| # | Sintoma observado | Causa raiz |
|---|---|---|
| 1 | Pós-processamento travava silenciosamente por 1h+ durante inferência de ensemble | XLA CPU JIT no `model.predict()` + `tf.data` interno ignorando `run_functions_eagerly` |
| 2 | Supervisor recebia SIGTERM sem motivo após poucas horas | `vmIdleTimeout` padrão do WSL2 desligava a VM por idle |
| 3 | Serviço (user ou system) era morto cada ~1 min sem intervenção | Distribuição Ubuntu do WSL desligava quando a última conexão Plan 9 fechava |

---

## 2. Arquitetura final

```
Windows
 └─ powershell Start-Process wsl.exe -u root sleep infinity  (PID Windows, WindowStyle Hidden)
     │
     └─ mantém conexão P9 aberta → impede WSL de desligar a distribuição
         │
         WSL Ubuntu (VM com vmIdleTimeout=-1)
          ├─ systemd (PID 1, system slice)
          │   └─ treino-rerun5.service  (Restart=on-failure, User=guga)
          │       ├─ bash run_wsl_gpu_reruns.sh
          │       │   └─ python run_rerun_queue.py
          │       │       └─ python run_all_experiments_supervisor.py  (stall_timeout=3600s)
          │       │           └─ python run_all_experiments.py
          │       │               └─ python run_all_efficientnetv2b0_variants.py  (variante isolada)
          │       │                   └─ GPU TensorFlow training / CPU ensemble inference
          │       │
          │       └─ logs em /mnt/c/Users/ghmd1/Desktop/treinamento/rerun_logs/
          │
          └─ outputs em /mnt/a/outputs/  (via symlink treinamento/outputs → /a/outputs)
```

---

## 3. Correções aplicadas em ordem

### 3.1. XLA CPU JIT travando pós-processamento

**Sintoma:** durante `[posprocess] Split 'validacao': inferencia do fold 1/5` aparecia warning `Very slow compile? [Compiling module a_inference_one_step_on_data__ for CPU]` e o processo ficava parado sem escrever log. Supervisor matava após 3600s (stall_timeout).

**Causa:** `model.predict()` chamado com `tf.device("/CPU:0")` dispara internamente um `tf.data.Dataset` que aciona XLA CPU JIT mesmo com `tf.config.optimizer.set_jit(False)` e `tf.config.run_functions_eagerly(True)` (o warning do TF confirma: *"this option does not apply to tf.data functions"*).

**Correção:** editado `efficientnetv2b0_kfold_runner.py`:

- Adicionado antes do `evaluate_ensemble()` no loop pós-proc:
  ```python
  tf.config.run_functions_eagerly(True)
  ```
- Substituído `model.predict(...)` por **loop numpy puro** na função `evaluate_ensemble`:
  ```python
  with tf.device("/CPU:0"):
      model = tf.keras.models.load_model(model_path)
      if eval_inputs is not None:
          batches = [
              model(eval_inputs[i : i + config.batch_size], training=False).numpy()
              for i in range(0, len(eval_inputs), config.batch_size)
          ]
          probabilities = np.concatenate(batches, axis=0)
      else:
          batches = [model(batch[0], training=False).numpy() for batch in dataset]
          probabilities = np.concatenate(batches, axis=0)
  ```
- Adicionado heartbeats (`print(..., flush=True)`) em cada etapa do pós-proc (fold_results, curvas, ensemble, Grad-CAM, artifacts_summary) para o supervisor conseguir detectar atividade.

**Arquivo afetado:** `efficientnetv2b0_kfold_runner.py`

### 3.2. WSL VM desligando por idle

**Sintoma:** processo morria depois de 2–8 horas com `Supervisor recebeu sinal 15`.

**Causa:** `vmIdleTimeout` padrão (em milissegundos) desliga a VM inteira quando não há atividade.

**Correção:** criar `C:\Users\ghmd1\.wslconfig`:

```ini
[wsl2]
vmIdleTimeout=-1
```

Aplicar com `wsl --shutdown` (uma vez) — próximos `wsl` comandos sobem com o novo config.

### 3.3. Distribuição Ubuntu desligando quando a última conexão P9 fecha

**Sintoma:** mesmo com `vmIdleTimeout=-1`, o serviço morria em ~1–2 minutos. No `journalctl` aparecia:
```
Operation canceled @p9io.cpp:258 (AcceptAsync)
systemd-logind[193]: The system will power off now!
systemd-logind[193]: System is powering down.
```

**Causa:** o `vmIdleTimeout` protege a **VM**, mas cada **distribuição** (Ubuntu) é encerrada quando a última conexão Plan 9 (protocolo usado pelo WSL para ligar Windows ↔ Linux) fecha. Cada `wsl -d Ubuntu -- bash -c "..."` abre uma conexão, e ao terminar o comando a conexão cai. Se não houver outra aberta, a distro desliga graciosamente.

**Correção:** manter uma conexão P9 permanente via processo Windows em background:

```powershell
Start-Process -FilePath 'wsl.exe' -ArgumentList '-d Ubuntu -u root -- sleep infinity' -WindowStyle Hidden
```

Isso cria um `wsl.exe` escondido no Windows que mantém `sleep infinity` rodando dentro do Ubuntu. Enquanto esse processo existir, o WSL considera a distro ativa e não a desliga.

### 3.4. Serviço systemd robusto no system slice

**Tentativas intermediárias que não resolveram:**
- `loginctl enable-linger guga` — só protege `user@1000.service` quando sessão de login fecha; não ajuda contra distro shutdown
- `KillUserProcesses=no` em `logind.conf` — só afeta fim de sessão, não desligamento da distro
- Serviço user em `~/.config/systemd/user/` — dependia de `user@1000.service`

**Correção final:** serviço system-level em `/etc/systemd/system/treino-rerun5.service` rodando como user `guga`, independente de sessões:

```ini
[Unit]
Description=Treinamento rerun5 EfficientNetV2B0
After=network.target

[Service]
Type=simple
User=guga
Group=guga
WorkingDirectory=/mnt/c/Users/ghmd1/Desktop/treinamento
ExecStart=/usr/bin/bash /mnt/c/Users/ghmd1/Desktop/treinamento/run_wsl_gpu_reruns.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:/mnt/c/Users/ghmd1/Desktop/treinamento/rerun_logs/rerun_queue_launch.log
StandardError=append:/mnt/c/Users/ghmd1/Desktop/treinamento/rerun_logs/rerun_queue_launch.log
Environment=HOME=/home/guga

[Install]
WantedBy=multi-user.target
```

Comandos para registrar:
```bash
sudo systemctl daemon-reload
sudo systemctl enable treino-rerun5.service
sudo systemctl start treino-rerun5.service
```

### 3.5. Stall timeout do supervisor

**Ajuste:** em `run_rerun_queue.py`, o item `rerun5_from_scratch_unfrozen_adamw` recebeu `stall_timeout_seconds: 3600`. O default era 900s (15 min), que matava o pós-processamento na transição entre folds (principalmente durante o build da fonte do Matplotlib em primeira execução).

### 3.6. Outputs no disco A:

Para não consumir o disco C:, os outputs foram movidos para `A:\outputs`. Dentro do projeto existe um symlink:

```
/mnt/c/Users/ghmd1/Desktop/treinamento/outputs -> /a/outputs
```

Em WSL esse symlink é resolvido corretamente para `A:\outputs` via NTFS. Criar com Git Bash:

```bash
ln -s /a/outputs /c/Users/ghmd1/Desktop/treinamento/outputs
```

---

## 4. Como iniciar do zero (máquina já configurada)

Depois de reboot ou após `wsl --shutdown`:

### 4.1. Iniciar keepalive do WSL (Windows PowerShell ou qualquer terminal):

```bash
powershell.exe -Command "Start-Process -FilePath 'wsl.exe' -ArgumentList '-d Ubuntu -u root -- sleep infinity' -WindowStyle Hidden"
```

### 4.2. Iniciar o serviço de treino:

```bash
wsl -d Ubuntu -u root -- systemctl start treino-rerun5.service
```

Se o serviço estiver `enabled`, ele sobe automaticamente com a distro e você só precisa garantir que o keepalive exista.

---

## 5. Como verificar saúde

### 5.1. Status do serviço:

```bash
wsl -d Ubuntu -u root -- systemctl status treino-rerun5.service
```

Esperado: `Active: active (running)`, `NRestarts=0`.

### 5.2. Keepalive Windows:

```bash
powershell.exe -Command "Get-Process wsl"
```

Deve haver pelo menos 2 `wsl.exe` (o keepalive em Hidden + o do comando atual).

### 5.3. Log do treino:

```bash
tail -20 /c/Users/ghmd1/Desktop/treinamento/rerun_logs/rerun5_from_scratch_unfrozen_adamw_child_windows.log
```

### 5.4. GPU:

```bash
wsl -d Ubuntu -- nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
```

**Interpretação:**
- **80–100%** com 4–6 GB → treino de fold rodando na GPU
- **0–15%** com 1–2 GB → pós-processamento CPU (normal pela correção 3.1)
- **0%** com 0 GB por >5 min → investigar, pode estar travado

### 5.5. Supervisor log (se desconfiar):

```bash
tail -15 /c/Users/ghmd1/Desktop/treinamento/rerun_logs/rerun5_from_scratch_unfrozen_adamw_supervisor_windows.log
```

Qualquer "Supervisor recebeu sinal 15" recente é sinal de que o serviço foi interrompido.

---

## 6. Como recuperar após uma falha

### Cenário A: Serviço parou mas WSL ainda vivo

```bash
wsl -d Ubuntu -u root -- systemctl start treino-rerun5.service
```

### Cenário B: WSL inteiro caiu

```bash
powershell.exe -Command "Start-Process -FilePath 'wsl.exe' -ArgumentList '-d Ubuntu -u root -- sleep infinity' -WindowStyle Hidden"
wsl -d Ubuntu -u root -- systemctl start treino-rerun5.service
```

### Cenário C: Pós-processamento travou em alguma variante

Verifique o log. Se for a mensagem `Very slow compile? [Compiling module ... for CPU]`, significa que o fix 3.1 não foi aplicado ou foi revertido — confirme que `efficientnetv2b0_kfold_runner.py` tem `tf.config.run_functions_eagerly(True)` antes do `evaluate_ensemble` e o loop numpy na função.

---

## 7. Arquivos e recursos criados/modificados

| Arquivo | Propósito |
|---|---|
| `C:\Users\ghmd1\.wslconfig` | Desativa idle timeout da VM (`vmIdleTimeout=-1`) |
| `/etc/systemd/system/treino-rerun5.service` | Serviço system-level que executa o treino como user `guga` |
| `/etc/systemd/logind.conf.d/keep-user-processes.conf` | `KillUserProcesses=no` (precaução extra) |
| `efficientnetv2b0_kfold_runner.py` | Loop numpy em `evaluate_ensemble`, heartbeats, `run_functions_eagerly(True)` |
| `run_rerun_queue.py` | `stall_timeout_seconds: 3600` para rerun5 |
| `outputs → /a/outputs` | Symlink para salvar no disco A: |

### Recursos em runtime:

| Item | Onde |
|---|---|
| Keepalive WSL | Windows: `wsl.exe` com `-u root sleep infinity` (Hidden) |
| Linger user guga | `loginctl enable-linger guga` (defensivo, desnecessário com system service) |
| Cron de monitoramento | Sessão Claude (expira em 7 dias) |

---

## 8. Checklist pré-flight

Antes de lançar um novo rerun:

- [ ] `.wslconfig` com `vmIdleTimeout=-1` em `C:\Users\<user>\.wslconfig`
- [ ] Keepalive `wsl.exe sleep infinity` rodando em Windows hidden
- [ ] Serviço `treino-rerun5.service` registrado em `/etc/systemd/system/` e `enabled`
- [ ] `efficientnetv2b0_kfold_runner.py` com `run_functions_eagerly(True)` e loop numpy em `evaluate_ensemble`
- [ ] Symlink `outputs` → `/a/outputs` criado
- [ ] Disco A: com ≥ 30 GB livres
- [ ] `nvidia-smi` responde dentro do WSL e mostra a RTX 4060 Ti

Com isso, o treino roda do começo ao fim sem intervenção.
