<!-- slide: dark -->
<div class="title-wrap">
<div class="eyebrow">AI403 Deep Learning · Final Project #13</div>
<h1>Multi-horizon Electricity<br>Load Forecasting</h1>
<p class="sub">Day-ahead forecasting of the PJM East grid with LSTM, Transformer and Temporal Fusion Transformer</p>
<div class="bar"></div>
<div class="team-name">TEAM FORECAST_AI</div>
<div class="team">
<span>Mohab Mohamed Moustafa Mahmoud Mohamed Aboklela</span><span class="id">202303375</span>
<span>Rosana Zarif Sadek Semean</span><span class="id">202303306</span>
<span>Yousef Mohamed Ragab Abdo AboEita</span><span class="id">202302137</span>
<span>Avronia Magdy Tawfik Boutros</span><span class="id">202303908</span>
<span>Sandra Samer Samir Fargallah</span><span class="id">202303334</span>
</div>
<div class="meta">Pharos University in Alexandria · Faculty of Computer Science &amp; AI · Fall 2026 · Instructor: Dr. Sahar Ghanem</div>
</div>

---

<div class="kicker">01 · Problem &amp; motivation</div>
<h2>Forecast tomorrow's 24 hourly loads, every day at midnight</h2>
<div class="body">
<div class="cols wide-left">
<div>
<ul>
<li><b>Task:</b> day-ahead, multi-horizon forecasting of PJM East load (≈ 32 GW average, USA)</li>
<li><b>Input:</b> last 168 h of load + 10 calendar features; the calendar of the next 24 h is known in advance</li>
<li><b>Output:</b> 24 hourly loads in <b>one forward pass</b>, so errors cannot compound hour by hour</li>
<li><b>Why it matters:</b> 1 % less error ≈ 320 MW every hour, the output of a mid-size gas turbine</li>
<li><b>Metrics:</b> MAE (primary), RMSE, MAPE and MAE per horizon; 3 seeds + bootstrap CIs</li>
</ul>
</div>
<div>
<table class="t">
<tr><th>#</th><th>Hypothesis</th></tr>
<tr><td><b>H1</b></td><td>The best deep model beats the baselines</td></tr>
<tr><td><b>H2</b></td><td>Calendar features help, most on weekends and holidays</td></tr>
<tr><td><b>H3</b></td><td>The Transformer beats the LSTM at late horizons (h13–24)</td></tr>
<tr><td><b>H4</b></td><td>Pretraining on other regions beats training from scratch</td></tr>
</table>
</div>
</div>
<div class="takeaway">One forecast per day at 00:00 → <b>24 hourly values</b>, scored on <b>579 days the models never saw</b>.</div>
</div>

---

<div class="kicker">02 · Data</div>
<h2>16 years of hourly load, split strictly by time</h2>
<div class="body">
<div class="kpis">
<div class="kpi"><div class="num">145k</div><div class="lbl">hourly rows, 2002 – 2018 (public, CC0)</div></div>
<div class="kpi"><div class="num blue">113,760</div><div class="lbl">training windows (train 2002 – 2014)</div></div>
<div class="kpi"><div class="num">731</div><div class="lbl">validation days (2015 – 2016) for tuning and early stopping</div></div>
<div class="kpi"><div class="num teal">579</div><div class="lbl">test days (2017 – Aug 2018), used once at the end</div></div>
</div>
<div class="fig-s">{{fig:experiments/eda/fig1_series_split.png|Chronological split: the model only ever learns from the past}}</div>
<p class="small">Cleaning: sort, fix daylight-saving duplicates and gaps · scaler fitted on training data only · real extremes (heatwaves, Hurricane Sandy) kept · a window belongs to a split only if all 24 targets are inside it — asserted by <code>check_split.py</code></p>
</div>

---

<div class="kicker">03 · Method</div>
<h2>Five model families, one pipeline, one training protocol</h2>
<div class="body">
<div class="cols three grow">
<div class="card grey"><div class="tag">Baselines</div><h3>Seasonal naive · LightGBM</h3><p>"Same hour last week / yesterday", and gradient-boosted trees on 20 lag, summary and calendar features.</p></div>
<div class="card orange"><div class="tag">Classic</div><h3>LSTM encoder–decoder</h3><p>Encoder reads 168 h into a memory state; the decoder turns it plus the future calendar into 24 loads. 40k params.</p></div>
<div class="card"><div class="tag">Modern</div><h3>Patch Transformer</h3><p>7 daily patches; each future hour attends to the past days. <b>Attention written by hand.</b> 187k params.</p></div>
<div class="card teal"><div class="tag">Advanced</div><h3>Temporal Fusion Transformer</h3><p>Variable selection, gated residual networks, LSTM and interpretable attention. 44k params.</p></div>
<div class="card navy"><div class="tag">Transfer</div><h3>Pretrained Transformer</h3><p>Pretrained on 10 other PJM regions (up to 2014 only), then fine-tuned on PJM East.</p></div>
<div class="card grey"><div class="tag">Shared protocol</div><h3>Fair comparison</h3><p>AdamW · L1 loss · batch 256 · ≤ 60 epochs + early stopping · same LR grid · <b>RevIN</b> · 3 seeds · 49 runs, CPU only.</p></div>
</div>
</div>

---

<div class="kicker">03 · Method</div>
<h2>Inside the patch Transformer</h2>
<div class="body">
<div>
<div class="lane">Encoder — what happened last week</div>
<div class="flow">
<div class="box"><b>Past week</b>168 h × 11 features</div><div class="arrow">→</div>
<div class="box"><b>RevIN</b>normalise by the window's own mean/SD</div><div class="arrow">→</div>
<div class="box"><b>Patch embedding</b>7 daily tokens × 64</div><div class="arrow">→</div>
<div class="box"><b>+ Positional encoding</b>sinusoidal</div><div class="arrow">→</div>
<div class="box hl"><b>2 × encoder block</b>self-attention between days</div>
</div>
</div>
<div>
<div class="lane">Decoder — what tomorrow looks like</div>
<div class="flow">
<div class="box"><b>24 hour queries</b>calendar + hour embedding</div><div class="arrow">→</div>
<div class="box hl"><b>Self-attention</b>between the 24 hours</div><div class="arrow">→</div>
<div class="box hl"><b>Cross-attention</b>each hour looks at the 7 days</div><div class="arrow">→</div>
<div class="box"><b>Linear head</b>+ RevIN inverse</div><div class="arrow">→</div>
<div class="box out"><b>24 loads</b>in MW</div>
</div>
</div>
<div class="cols">
<div class="card"><h3>Attention, written from the equations</h3><p>softmax(QKᵀ / √d<sub>k</sub>) · V with 4 heads of 16 dims, head split/merge, positional encoding, pre-norm blocks — no <code>nn.MultiheadAttention</code> or <code>nn.Transformer</code>.</p></div>
<div class="card teal"><h3>Why daily patches?</h3><p>One token = one whole day, so attention compares days directly ("is tomorrow like last Tuesday?") on a 7×7 instead of 168×168 matrix.</p></div>
</div>
<div class="takeaway">187k parameters, yet the <b>fastest deep model to train (≈ 9 min on one CPU thread)</b>: the 7 day-tokens are processed in parallel, while the LSTM and TFT step through 168 hours one by one.</div>
</div>

---

<div class="kicker">04 · Results</div>
<h2>The Transformer is the most accurate model</h2>
<div class="body">
<div class="kpis">
<div class="kpi"><div class="num blue">1,389 MW</div><div class="lbl">Transformer test MAE (± 12 over 3 seeds)</div></div>
<div class="kpi"><div class="num">4.28 %</div><div class="lbl">MAPE on 579 unseen test days</div></div>
<div class="kpi"><div class="num teal">−51 MW</div><div class="lbl">vs LightGBM, 95 % CI [−98, −4]</div></div>
<div class="kpi"><div class="num orange">−60 %</div><div class="lbl">error vs "same hour last week"</div></div>
</div>
<table class="t">
<tr><th>Model</th><th class="n">Test MAE (MW)</th><th class="n">95 % CI</th><th class="n">RMSE (MW)</th><th class="n">MAPE (%)</th></tr>
<tr class="base"><td>Seasonal naive (last week)</td><td class="n">3,511</td><td class="n">[3,289, 3,735]</td><td class="n">4,794</td><td class="n">10.99</td></tr>
<tr class="base"><td>Seasonal naive (yesterday)</td><td class="n">2,300</td><td class="n">[2,179, 2,422]</td><td class="n">3,134</td><td class="n">7.32</td></tr>
<tr><td>LightGBM (baseline)</td><td class="n">1,440 ± 3</td><td class="n">[1,362, 1,519]</td><td class="n">2,109</td><td class="n">4.45</td></tr>
<tr><td>LSTM enc-dec (classic)</td><td class="n">1,460 ± 43</td><td class="n">[1,392, 1,531]</td><td class="n">2,067</td><td class="n">4.54</td></tr>
<tr><td>TFT (advanced)</td><td class="n">1,432 ± 28</td><td class="n">[1,362, 1,508]</td><td class="n">2,053</td><td class="n">4.43</td></tr>
<tr><td>Transformer, pretrained + fine-tuned</td><td class="n">1,394 ± 17</td><td class="n">[1,322, 1,471]</td><td class="n">2,037</td><td class="n">4.31</td></tr>
<tr class="best"><td>Transformer (modern)</td><td class="n">1,389 ± 12</td><td class="n">[1,316, 1,467]</td><td class="n">2,036</td><td class="n">4.28</td></tr>
</table>
</div>

---

<div class="kicker">04 · Results</div>
<h2>Where in the day — and in the year — the models struggle</h2>
<div class="body">
<div class="cols wide-left grow">
<div class="fig-m">{{fig:experiments/figures/per_horizon_mae.png|Test MAE for each hour of the forecast day}}</div>
<div>
<ul>
<li>Error peaks at the <b>evening peak (18:00)</b>, not at hour 24</li>
<li>The Transformer gains most in the <b>early hours</b>: 827 vs 959 MW (LSTM) for h1–12</li>
<li>From midday on, all models converge to ≈ 1,950 MW — they hit the same wall</li>
<li><b>Summer is hardest</b> for every model (Transformer 1,779 MW vs 1,216 MW in spring)</li>
</ul>
<div class="takeaway" style="margin-top:5mm">The limit is <b>missing weather information</b>, not the architecture.</div>
</div>
</div>
</div>

---

<div class="kicker">04 · Model comparison</div>
<h2>Hypothesis verdicts</h2>
<div class="body">
<table class="t">
<tr><th>Hypothesis</th><th>Test (paired bootstrap over test days)</th><th class="n">Δ MAE (MW)</th><th class="n">95 % CI</th><th>Verdict</th></tr>
<tr><td><b>H1</b> Deep model beats the baselines</td><td>Transformer − LightGBM</td><td class="n">−51</td><td class="n">[−98, −4]</td><td><span class="badge ok">Supported</span></td></tr>
<tr><td><b>H2</b> Calendar helps, most on weekends/holidays</td><td>No calendar − full (weekend+holiday / weekday)</td><td class="n">+64 / +16</td><td class="n">[−5, +67]</td><td><span class="badge mid">Inconclusive</span></td></tr>
<tr><td><b>H3</b> Transformer wins at late horizons</td><td>Transformer − LSTM (h13–24 / h1–12)</td><td class="n">−11 / −132</td><td class="n">[−47, +27]</td><td><span class="badge mid">Inconclusive</span></td></tr>
<tr><td><b>H4</b> Pretraining helps — 13 years of data</td><td>Fine-tuned − scratch</td><td class="n">+5</td><td class="n">[−9, +20]</td><td><span class="badge mid">Inconclusive</span></td></tr>
<tr><td><b>H4</b> Pretraining helps — 2 years of data</td><td>Fine-tuned − scratch</td><td class="n">−193</td><td class="n">[−223, −163]</td><td><span class="badge ok">Supported</span></td></tr>
</table>
<div class="cols">
<div class="card"><h3>What the inconclusive results tell us</h3><p>H2 goes in the predicted direction (4× larger effect on weekends and holidays) but the CI touches 0. H3: the Transformer's advantage is in the <b>early</b> hours, not the late ones.</p></div>
<div class="card teal"><h3>Method</h3><p>Errors are compared <b>on the same day</b> for both models, cancelling out how hard each day is; 2,000 bootstrap resamples of the 579 test days.</p></div>
</div>
<div class="takeaway">Deep learning wins (H1) and pretraining wins when data is scarce (H4); calendar and horizon effects are <b>real but small</b>.</div>
</div>

---

<div class="kicker">05 · Experiments</div>
<h2>Hyperparameter and data-representation studies</h2>
<div class="body">
<div class="cols">
<div>
<div class="lane">Same learning-rate grid for every model · best val MAE in 30 epochs</div>
<table class="t">
<tr><th>Model</th><th class="n">3e-4</th><th class="n">1e-3</th><th class="n">3e-3</th></tr>
<tr><td>Transformer</td><td class="n">1,456</td><td class="n"><b>1,265</b></td><td class="n">1,269</td></tr>
<tr><td>LSTM</td><td class="n">1,466</td><td class="n">1,351</td><td class="n"><b>1,275</b></td></tr>
<tr><td>TFT</td><td class="n">1,432</td><td class="n">1,313</td><td class="n"><b>1,249</b></td></tr>
</table>
<p class="small" style="margin-top:3mm">Transformer width d = 32 / 64 / 128 → 1,368 / 1,265 / 1,213 · dropout 0 / 0.1 / 0.3 → <b>1,176</b> / 1,265 / 1,512</p>
</div>
<div>
<div class="lane">Representation · Transformer, 3 seeds</div>
<table class="t">
<tr><th>Variant</th><th class="n">Val MAE</th><th class="n">Test MAE</th></tr>
<tr><td>Cyclical sin/cos + RevIN (final)</td><td class="n">1,205</td><td class="n">1,389</td></tr>
<tr><td>Without RevIN</td><td class="n">1,331</td><td class="n">1,520</td></tr>
<tr class="best"><td>One-hot calendar</td><td class="n">1,182</td><td class="n">1,340</td></tr>
</table>
<p class="small" style="margin-top:3mm">Lookback 72 h / 336 h: 1,209 / 1,225 val MAE — no better than 168 h</p>
</div>
</div>
<div class="cols three">
<div class="card orange"><h3>Learning rate</h3><p>3e-4 is too slow for every model; 3e-3 helps the LSTM and TFT most. All models got the <b>same grid</b>.</p></div>
<div class="card"><h3>Capacity &amp; regularisation</h3><p>Wider helps (d = 128: −52 MW, at 3.8× the parameters). No dropout is best: with 114k windows the model is not over-fitting.</p></div>
<div class="card teal"><h3>Representation</h3><p>One-hot gives every hour its <b>own weights</b>; sin/cos forces a smooth wave. One week (168 h) is the right lookback.</p></div>
</div>
<div class="takeaway">Best configuration found: <b>one-hot calendar Transformer — 1,340 MW, MAPE 4.13 %</b>, 99 MW better than LightGBM (selected on validation).</div>
</div>

---

<div class="kicker">05 · Ablation &amp; transfer learning</div>
<h2>What each component is really worth</h2>
<div class="body">
<div class="cols wide-left">
<div>
<table class="t">
<tr><th>Component removed</th><th class="n">Test MAE</th><th class="n">Δ vs full</th></tr>
<tr><td>— full Transformer</td><td class="n">1,389</td><td class="n"></td></tr>
<tr class="best"><td>RevIN normalisation</td><td class="n">1,520</td><td class="n">+131 (+9.5 %)</td></tr>
<tr><td>Calendar features</td><td class="n">1,420</td><td class="n">+31 (+2.2 %)</td></tr>
<tr><td>Decoder cross-attention</td><td class="n">1,380</td><td class="n">−9 (noise)</td></tr>
<tr><td>TFT attention layer (TFT: 1,432)</td><td class="n">1,417</td><td class="n">−15 (noise)</td></tr>
</table>
<p class="small" style="margin-top:3mm">Calendar removal hurts 4× more on weekends and holidays (+64) than on weekdays (+16).</p>
</div>
<div>
<div class="lane">Pretrained on 10 regions vs scratch · test MAE</div>
<div class="kpis" style="grid-template-columns:1fr 1fr">
<div class="kpi"><div class="num">+5 MW</div><div class="lbl">13 years of PJM data — no gain</div></div>
<div class="kpi"><div class="num teal">−12 %</div><div class="lbl">2 years of data: 1,608 → 1,415 MW</div></div>
</div>
<p class="small" style="margin-top:3mm">Zero-shot (never saw PJM East): 1,546 MW — already better than both naive rules.</p>
</div>
</div>
<div class="cols three">
<div class="card navy"><h3>RevIN</h3><p>Average load in 2017–18 is lower than in 2002–14. Normalising each window by its own statistics removes this <b>distribution shift</b>.</p></div>
<div class="card"><h3>Attention</h3><p>Removing it in either model changes nothing measurable: patches, LSTM and gating already capture a one-day horizon.</p></div>
<div class="card teal"><h3>Transfer</h3><p>With 2 years of data, pretraining nearly matches a model trained on 13 years.</p></div>
</div>
<div class="takeaway">Components that encode <b>prior knowledge</b> (normalisation, calendar) matter more than attention; pretraining is a <b>data-efficiency</b> tool.</div>
</div>

---

<div class="kicker">06 · Error analysis</div>
<h2>Where and why the models fail</h2>
<div class="body">
<div class="cols wide-right grow">
<div>
<ul>
<li><b>8 of the 10 worst days</b>: daily load jumped &gt; 2.5 GW — mostly <b>a heat spell breaking overnight</b>, peak over-forecast by up to 11 GW (6 Jul 2018)</li>
<li><b>First day of a heatwave</b>: peaks under-forecast by 1.7 GW on the hottest 10 % of days</li>
<li><b>Holidays</b>: one bit cannot separate Veterans Day (businesses open) from Christmas</li>
<li>LightGBM fails on the same weather days</li>
</ul>
</div>
{{fig:experiments/figures/worst_days.png|The four worst test days: actual (black), Transformer (blue), LightGBM (grey)}}
</div>
<div class="takeaway">Root cause: <b>tomorrow's temperature is not in the input</b> — no load-only model can foresee these days.</div>
</div>

---

<div class="kicker">07 · Conclusion</div>
<h2>Conclusion and future work</h2>
<div class="body">
<div class="cols">
<div class="card"><h3>What we showed</h3>
<ul>
<li>A leakage-free, reproducible day-ahead forecaster: <b>4.28 % MAPE</b>, <b>4.13 %</b> with a one-hot calendar</li>
<li>Deep learning beats a strong LightGBM baseline — <b>significantly but modestly</b></li>
<li><b>Inductive bias over size</b>: daily patches, RevIN and known calendar inputs matter most</li>
<li>Pretraining pays off when target data is <b>scarce</b></li>
</ul>
</div>
<div class="card teal"><h3>Next steps</h3>
<ul>
<li>Add <b>temperature forecasts</b> as known future inputs</li>
<li><b>Probabilistic</b> forecasts with quantile loss for reserve planning</li>
<li>Richer calendar: holiday types and bridge days</li>
<li>Ensemble the Transformer with LightGBM; retrain every model at its best screened setting</li>
</ul>
</div>
</div>
<div class="takeaway">The remaining error is <b>weather, not model capacity</b>.</div>
</div>

---

<!-- slide: dark -->
<div class="closing">
<div class="title-wrap" style="height:auto">
<div class="eyebrow">Team forecast_AI</div>
</div>
<h1>Thank you</h1>
<p>Questions?</p>
</div>
