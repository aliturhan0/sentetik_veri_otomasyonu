document.addEventListener('DOMContentLoaded', () => {
    const API = 'http://127.0.0.1:8000';
    const $ = id => document.getElementById(id);
    let currentFile = null, f1Chart = null, distilledFile = null;
    
    // Set absolute download links
    $('dl-distilled').href = API + '/api/download_distilled';
    document.querySelector('#dl-card .btn-download').href = API + '/api/download_generated';

    function getDesktopApi() {
        return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
    }

    async function saveCsvViaDesktop(kind) {
        const method = kind === 'distilled' ? 'save_distilled_csv' : 'save_generated_csv';
        const desktopApi = getDesktopApi();
        if (!desktopApi || !desktopApi[method]) {
            log('Masaüstü kaydetme servisi hazır değil. Uygulamayı main.py ile açın.', 'err');
            return false;
        }

        const result = await desktopApi[method]();
        if (result && result.ok) {
            log('CSV kaydedildi: ' + result.path, 'ok');
        } else if (result && !result.cancelled) {
            log(result.message || 'CSV kaydedilemedi.', 'err');
        }
        return true;
    }

    $('dl-distilled').addEventListener('click', async e => {
        if (getDesktopApi()) {
            e.preventDefault();
            await saveCsvViaDesktop('distilled');
        }
    });
    document.querySelector('#dl-card .btn-download').addEventListener('click', async e => {
        if (getDesktopApi()) {
            e.preventDefault();
            await saveCsvViaDesktop('generated');
        }
    });

    // === NAV ===
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            item.classList.add('active');
            $('panel-' + item.dataset.panel).classList.add('active');
            $('panel-title').textContent = item.querySelector('span').textContent;
            if (item.dataset.panel === 'analysis' && f1Chart) {
                requestAnimationFrame(() => f1Chart.update());
            }
        });
    });

    // === CLOCK ===
    setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString('tr-TR',{hour12:false}); }, 1000);

    // === LOG ===
    function log(msg, type='') {
        const d = document.createElement('div');
        d.className = 'clog ' + type;
        const t = new Date().toLocaleTimeString('tr-TR',{hour12:false});
        d.innerHTML = `<span style="opacity:.4">[${t}]</span> ${msg}`;
        $('console-out').appendChild(d);
        $('console-out').scrollTop = 99999;
    }

    function fmtCount(value, fallback='—') {
        if (value === null || value === undefined || value === '') return fallback;
        return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : fallback;
    }

    let statusOkOnce = false;
    let statusWaitingLogged = false;
    let previousStatusKey = '';
    let latestAnalysisRestored = false;

    function createFallbackChart(canvas) {
        const chart = {
            data: {
                labels: ['Baseline', 'Augmented'],
                datasets: [{data: [0, 0]}],
            },
            update() {
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                canvas.width = Math.max(1, Math.floor(rect.width * dpr));
                canvas.height = Math.max(1, Math.floor(rect.height * dpr));
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                const w = rect.width || 420, h = rect.height || 260;
                ctx.clearRect(0, 0, w, h);
                ctx.fillStyle = '#0f1724';
                ctx.fillRect(0, 0, w, h);
                const values = this.data.datasets[0].data.map(v => Math.max(0, Math.min(1, Number(v) || 0)));
                const labels = this.data.labels;
                const colors = ['#6366f1', '#10b981', '#22d3ee', '#f59e0b'];
                const barW = Math.max(34, Math.min(84, w / (values.length * 2.5)));
                values.forEach((v, i) => {
                    const x = w * ((i + 1) / (values.length + 1)) - barW / 2;
                    const bh = v * (h - 70);
                    const y = h - 38 - bh;
                    ctx.fillStyle = colors[i];
                    ctx.fillRect(x, y, barW, bh);
                    ctx.fillStyle = '#dbeafe';
                    ctx.font = '700 13px system-ui';
                    ctx.textAlign = 'center';
                    ctx.fillText(v.toFixed(3), x + barW / 2, Math.max(18, y - 8));
                    ctx.fillStyle = '#94a3b8';
                    ctx.font = '12px system-ui';
                    ctx.fillText(labels[i] || '', x + barW / 2, h - 14);
                });
            },
        };
        chart.update();
        window.addEventListener('resize', () => chart.update());
        return chart;
    }

    // === STATUS ===
    async function checkStatus() {
        try {
            const r = await fetch(API+'/api/system_status');
            const d = await r.json();
            statusOkOnce = true;
            if(d.model_loaded){
                $('sys-indicator').innerHTML='<span class="blink-dot green"></span>';
                $('sys-label').textContent='GODMODE Aktif';$('sys-label').style.color='var(--green)';
            } else {
                $('sys-indicator').innerHTML='<span class="blink-dot green"></span>';
                $('sys-label').textContent='Bağlandı';$('sys-label').style.color='var(--green)';
            }
            const seedText=fmtCount(d.seed_rows, d.seed_available ? 'Hazır' : 'Yok');
            $('sys-seed').textContent=seedText;
            const statusKey=[d.model_loaded,d.model_name,seedText].join('|');
            if(statusKey !== previousStatusKey) {
                log('Seed havuzu: '+seedText+(d.seed_rows == null ? '' : ' yörünge'),'ok');
                if(d.model_loaded) log('RCGAN modeli aktif: '+d.model_name,'ok');
                previousStatusKey=statusKey;
            }
        } catch(e) {
            $('sys-label').textContent=statusOkOnce ? 'İşlem Sürüyor' : 'Sunucu Hazırlanıyor';
            $('sys-label').style.color=statusOkOnce ? 'var(--amber)' : 'var(--amber)';
            if (!statusWaitingLogged) {
                log('Sunucu hazırlanıyor, bağlantı otomatik tekrar denenecek...', 'ok');
                statusWaitingLogged = true;
            }
        }
    }
    checkStatus();
    setInterval(checkStatus, 3000);

    // === CHART ===
    (function(){
        const ctx=$('f1Chart').getContext('2d');
        if (window.Chart) {
            Chart.defaults.color='#5a6578';Chart.defaults.font.family='Inter';
            f1Chart=new Chart(ctx,{type:'bar',data:{labels:['Baseline','Augmented'],datasets:[{data:[0,0],backgroundColor:['rgba(99,102,241,.45)','rgba(16,185,129,.6)'],borderColor:['#6366f1','#10b981'],borderWidth:2,borderRadius:6,barPercentage:.5}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{beginAtZero:true,max:1,grid:{color:'rgba(255,255,255,.03)'}},x:{grid:{display:false}}},plugins:{legend:{display:false}}}});
        } else {
            f1Chart = createFallbackChart($('f1Chart'));
        }
    })();

    async function restoreLatestAnalysis(attempt=0) {
        if(latestAnalysisRestored) return;
        try {
            const response = await fetch(API+'/api/latest_analysis');
            if(response.status === 404) return;
            if(!response.ok) throw new Error('Son analiz alınamadı.');
            const result = await response.json();
            showResults(result, true);
            latestAnalysisRestored = true;
            log('Son başarılı analiz sonucu geri yüklendi.','ok');
        } catch(e) {
            if(attempt < 20) setTimeout(()=>restoreLatestAnalysis(attempt+1), 500);
        }
    }
    // === SLIDER ===
    $('n-samples').addEventListener('input', e => { $('slider-val').textContent = e.target.value; });

    // === UPLOAD ===
    const dz=$('drop-zone');
    dz.addEventListener('click',()=>$('file-input').click());
    dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
    dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
    dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files.length)loadFile(e.dataTransfer.files[0])});
    $('file-input').addEventListener('change',e=>{if(e.target.files.length)loadFile(e.target.files[0])});
    $('btn-remove').addEventListener('click',()=>{
        currentFile=null;distilledFile=null;$('file-input').value='';
        dz.classList.remove('hidden');$('file-pill').classList.add('hidden');
        $('btn-generate').disabled=true;$('btn-distill').disabled=true;
        log('Dosya kaldırıldı.');
    });

    function loadFile(f) {
        if(!f.name.toLowerCase().endsWith('.csv')){
            log('Sadece CSV dosyası yüklenebilir. Seçilen dosya: '+f.name, 'err');
            return;
        }
        if(!f.size){
            log('Seçilen CSV dosyası boş. Lütfen veri içeren bir CSV seç.', 'err');
            return;
        }

        currentFile=f;
        $('fname').textContent=f.name;$('fsize').textContent=(f.size/1024).toFixed(0)+' KB';
        dz.classList.add('hidden');$('file-pill').classList.remove('hidden');
        $('btn-distill').disabled=false;
        $('btn-generate').disabled=false;  // generate also works without distill
        log('Dosya: '+f.name+' ('+(f.size/1024).toFixed(0)+' KB)','ok');
    }

    // === DISTILL ===
    $('btn-distill').addEventListener('click', async () => {
        if(!currentFile) return;
        const btn=$('btn-distill');
        btn.disabled=true; btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> Damıtılıyor...';
        $('distill-progress').classList.remove('hidden');
        log('Bilgi damıtma başlatıldı...','ok');

        let p=0;
        const iv=setInterval(()=>{p+=Math.random()*15;if(p>90)p=90;$('distill-thumb').style.width=p+'%'},300);

        try {
            const fd=new FormData(); fd.append('file',currentFile);
            const r=await fetch(API+'/api/distill',{method:'POST',body:fd});
            if(!r.ok) throw new Error('Damıtma hatası: '+r.status);
            const d=await r.json();
            clearInterval(iv);
            $('distill-thumb').style.width='100%';
            $('distill-text').textContent='Damıtma tamamlandı!';
            btn.innerHTML='<i class="fa-solid fa-check"></i> Damıtma Tamamlandı';

            // Build report
            const rpt=$('distill-report'); rpt.innerHTML='';
            const rep=d.report;

            // Original info
            addDistillRow(rpt,'Orijinal Veri', rep.original_rows.toLocaleString()+' satır, '+rep.original_cols+' sütun','');

            // Steps
            rep.steps.forEach(s=>{
                let val='', cls='';
                if(s.removed!==undefined){val='-'+s.removed;cls='removed';}
                else if(s.fixed!==undefined){val='~'+s.fixed+' düzeltildi';cls='fixed';}
                else {val=s.detail;cls='';}
                addDistillRow(rpt,s.name,val,cls);
                log('🔧 '+s.name+': '+s.detail);
            });

            // Summary
            const sum=document.createElement('div');
            sum.className='distill-summary';
            sum.textContent='✅ Temiz veri: '+rep.clean_rows.toLocaleString()+' satır ('+rep.reduction_pct+'% azaltma)';
            rpt.appendChild(sum);

            $('dl-distilled').classList.remove('hidden');
            log('Damıtma tamamlandı: '+rep.original_rows+' → '+rep.clean_rows+' satır ('+rep.reduction_pct+'% azaltma)','ok');

            // Store distilled file info for generate
            distilledFile = true;
        } catch(e) {
            clearInterval(iv); log(e.message,'err');
            btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-broom"></i> Veriyi Damıt';
        }
    });

    function addDistillRow(container, name, val, cls) {
        const r=document.createElement('div'); r.className='distill-row';
        r.innerHTML=`<span class="d-name">${name}</span><span class="d-val ${cls}">${val}</span>`;
        container.appendChild(r);
    }

    // === GENERATE ===
    $('btn-generate').addEventListener('click', async () => {
        if(!currentFile) return;
        const btn=$('btn-generate');
        const nSamples=parseInt($('n-samples').value);
        btn.disabled=true; btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> İşleniyor...';
        $('gen-progress').classList.remove('hidden');
        log('Sentez başlatıldı ('+nSamples+' örnek)...','ok');

        let p=0;
        const iv=setInterval(()=>{p+=Math.random()*10;if(p>92)p=92;$('gen-thumb').style.width=p+'%'},400);

        try {
            const fd=new FormData();
            fd.append('file',currentFile);
            fd.append('n_samples', nSamples.toString());
            const d = await new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open("POST", API+'/api/evaluate_pipeline', true);
                xhr.timeout = 1800000; // Büyük veri/CTGAN işlemleri için 30 dakika
                xhr.onload = function() {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try { resolve(JSON.parse(xhr.responseText)); } 
                        catch(e) { reject(new Error("JSON ayrıştırma hatası")); }
                    } else {
                        try {
                            let err = JSON.parse(xhr.responseText);
                            reject(new Error(err.detail || 'Hata: ' + xhr.status));
                        } catch(e) { reject(new Error('Hata: ' + xhr.status)); }
                    }
                };
                xhr.onerror = () => reject(new Error("Bağlantı koptu veya ağ hatası."));
                xhr.ontimeout = () => reject(new Error("Zaman aşımı (30 dakika aşıldı)."));
                xhr.send(fd);
            });
            if(d.error) { throw new Error(d.error); }
            clearInterval(iv);
            $('gen-thumb').style.width='100%';$('gen-text').textContent='Tamamlandı!';
            btn.innerHTML='<i class="fa-solid fa-check"></i> Tamamlandı';
            showResults(d);
        } catch(e) {
            clearInterval(iv); log(e.message,'err');
            btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-play"></i> Üretimi Başlat';
        }
    });

    function showResults(d, switchPanel=true) {
        const utilitySelection=d.utility && d.utility.protocol ? d.utility : null;
        const utilityRejected=utilitySelection && utilitySelection.accepted_for_utility === false;
        $('kpi-seed').textContent=d.seed_count.toLocaleString();
        $('kpi-gen').previousElementSibling.textContent=utilityRejected ? 'Üretilen Aday' : 'Üretilen';
        $('kpi-gen').textContent=d.gen_count.toLocaleString();
        $('kpi-factor').textContent='×'+d.multiplication_factor;
        $('kpi-cov').textContent='%'+d.generative_coverage;
        const imp=$('kpi-imp');

        const methodNames={rcgan:'RCGAN GODMODE',ctgan:'CTGAN (On-the-fly)',smote:'SMOTE+Gaussian'};
        const utilityAvailable=d.utility && d.utility.evaluable !== false;
        if(utilityAvailable) {
            $('kpi-score-label').textContent='F1 Score';
            $('kpi-f1').textContent=d.augmented_f1.toFixed(4);
            if(d.utility.f1_target_status === 'not_applicable_ceiling') {
                imp.textContent='TAVAN';
                imp.style.color='var(--cyan)';
            } else {
                imp.textContent=(d.improvement>0?'+':'')+d.improvement.toFixed(1)+'%';
                imp.style.color=d.improvement>0?'var(--green)':'var(--red)';
            }
            $('comparison-title').innerHTML='<i class="fa-solid fa-chart-column"></i> F1 Karşılaştırması';
            f1Chart.data.labels=['Seed F1',methodNames[d.method]||'Augmented F1'];
            f1Chart.data.datasets[0].data=[d.seed_f1,d.augmented_f1];
        } else if(d.quality_report && d.quality_report.components) {
            $('kpi-score-label').textContent='Kalite Skoru';
            $('kpi-f1').textContent=d.quality_report.overall_score.toFixed(1);
            imp.textContent=d.quality_report.grade;
            imp.style.color='var(--green)';
            const components=d.quality_report.components;
            $('comparison-title').innerHTML='<i class="fa-solid fa-chart-column"></i> Uygulanabilir Kalite Metrikleri';
            f1Chart.data.labels=['Benzerlik','Dağılım','Fizik'];
            f1Chart.data.datasets[0].data=[
                (components.fidelity?.score || 0)/100,
                (components.distribution?.score || 0)/100,
                (components.physical?.score || 0)/100
            ];
        } else {
            $('kpi-score-label').textContent='F1 Score';
            $('kpi-f1').textContent=d.augmented_f1.toFixed(4);
            imp.textContent=(d.improvement>0?'+':'')+d.improvement.toFixed(1)+'%';
            imp.style.color=d.improvement>0?'var(--green)':'var(--red)';
            $('comparison-title').innerHTML='<i class="fa-solid fa-chart-column"></i> F1 Karşılaştırması';
            f1Chart.data.labels=['Seed F1',methodNames[d.method]||'Augmented F1'];
            f1Chart.data.datasets[0].data=[d.seed_f1,d.augmented_f1];
        }
        f1Chart.update();

        // Details
        const dl=$('detail-list'); dl.innerHTML='';
        if(d.analysis_note){
            const r=document.createElement('div');r.className='detail-row';
            r.innerHTML=`<span class="label" style="color:var(--amber);font-weight:700">Analiz Notu</span><span class="val">${d.analysis_note}</span>`;
            dl.appendChild(r);
        }
        if(d.dataset_info){
            const di=d.dataset_info;
            const summaryRows=[];
            if(di.source_seed_rows) summaryRows.push(['Kaynak Seed Havuzu',di.source_seed_rows.toLocaleString()]);
            if(di.sampled_rows) summaryRows.push(['İşlenen Örnek',di.sampled_rows.toLocaleString()]);
            summaryRows.push(['Satır (temiz)',d.seed_count.toLocaleString()],['Özellik',di.features],['Sınıf',di.classes],
             ['Label','"'+di.label_col+'"'],['Format',di.is_waymo?'Waymo':'Genel'],
             ['Yöntem',methodNames[d.method]||d.method]
            );
            if(utilityAvailable) {
                summaryRows.push(['Seed F1',d.seed_f1.toFixed(4)],['Aug F1',d.augmented_f1.toFixed(4)]);
            } else {
                summaryRows.push(['F1 / Recall','Uygulanamaz (tek sınıflı seed)']);
            }
            summaryRows.forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
        }

        if(d.quality_report){
            const qr=d.quality_report;
            const hdrQ=document.createElement('div');hdrQ.className='detail-row';
            hdrQ.innerHTML='<span class="label" style="color:var(--cyan);font-weight:700">── Kalite Skoru ──</span><span class="val">'+qr.overall_score+'/100 ('+qr.grade+')</span>';
            dl.appendChild(hdrQ);
            [['Yönlendirme',qr.routing_explanation],
             ['Fidelity Skoru',qr.components?.fidelity?.score!=null?qr.components.fidelity.score+'/100':'Uygulanamaz'],
             ['Utility Skoru',qr.components?.utility?.score!=null?qr.components.utility.score+'/100':'Uygulanamaz'],
             ['Dağılım Skoru',qr.components?.distribution?.score!=null?qr.components.distribution.score+'/100':'Uygulanamaz'],
             ['Fizik Skoru',qr.components?.physical?.score!=null?qr.components.physical.score+'/100':'Uygulanamaz']
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
            if(qr.distribution_shift && qr.distribution_shift.warnings && qr.distribution_shift.warnings.length){
                qr.distribution_shift.warnings.slice(0,4).forEach(w=>{
                    const r=document.createElement('div');r.className='detail-row';
                    r.innerHTML=`<span class="label" style="color:var(--amber)">Dağılım Uyarısı</span><span class="val">${w.detail}</span>`;
                    dl.appendChild(r);
                });
            }
            if(qr.scientific_basis){
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML='<span class="label">Bilimsel Temel</span><span class="val">Fidelity + Utility + Distribution Shift + Fiziksel Tutarlılık</span>';
                dl.appendChild(r);
            }
        }

        if(d.generation_report && d.generation_report.rcgan_postprocess){
            const post=d.generation_report.rcgan_postprocess;
            const hdrG=document.createElement('div');hdrG.className='detail-row';
            hdrG.innerHTML='<span class="label" style="color:var(--cyan);font-weight:700">── RCGAN Üretim Kontrolü ──</span><span></span>';
            dl.appendChild(hdrG);
            const removed=post.diversity_filter?.removed || 0;
            const labels=Object.entries(post.anomaly_distribution_final || post.anomaly_distribution_after_filter || {})
                .map(([key,value])=>key+': '+Number(value).toLocaleString()).join(', ');
            const physicsRemoved=(d.generation_report.validator?.steps || [])
                .filter(step=>step.name === 'Fiziksel Validator')
                .reduce((total,step)=>total+(step.removed || 0),0);
            [['İstenen Yörünge',Number(post.requested_rows || d.gen_count).toLocaleString()],
             ['Üretilen Aday',Number(post.candidate_rows || post.requested_rows || d.gen_count).toLocaleString()],
             ['Fizik Onaylı Çıktı',Number(d.gen_count).toLocaleString()],
             ['Benzerlik Filtresi',removed ? removed.toLocaleString()+' tekrar-benzeri elendi' : 'Eleme yapılmadı'],
             ['Fiziksel Validator',physicsRemoved ? physicsRemoved.toLocaleString()+' zayıf yörünge elendi' : 'Tüm yörüngeler geçti'],
             ['Anomali Dağılımı',labels || '-']
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
        }

        // Distillation report in details
        if(d.distillation && d.distillation.steps){
            const hdr=document.createElement('div');hdr.className='detail-row';
            hdr.innerHTML='<span class="label" style="color:var(--cyan);font-weight:700">── Damıtma ──</span><span></span>';
            dl.appendChild(hdr);
            d.distillation.steps.forEach(s=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${s.name}</span><span class="val">${s.detail}</span>`;
                dl.appendChild(r);
            });
        }
        
        // Fidelity & Utility (Akademik Metrikler)
        if (d.fidelity && d.utility) {
            // Fidelity
            const hdrF=document.createElement('div');hdrF.className='detail-row';
            hdrF.innerHTML='<span class="label" style="color:var(--green);font-weight:700;margin-top:10px">── Fidelity (Benzerlik) ──</span><span></span>';
            dl.appendChild(hdrF);
            
            [['Cosine Benzerliği', (d.fidelity.cosine_similarity*100).toFixed(1)+'%'],
             ['Sütun Korelasyonu', (d.fidelity.column_correlation*100).toFixed(1)+'%']
            ].forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });

            // Utility
            const hdrU=document.createElement('div');hdrU.className='detail-row';
            hdrU.innerHTML='<span class="label" style="color:var(--yellow);font-weight:700;margin-top:10px">── Utility (Fayda) ──</span><span></span>';
            dl.appendChild(hdrU);
            
            let f1_target_str = !utilityAvailable ? 'Uygulanamaz'
                : d.utility.f1_target_status === 'not_applicable_ceiling' ? 'Tavan: artış mümkün değil'
                : d.utility.f1_target_met ? 'Başarılı' : 'Başarısız';
            let recall_target_str = !utilityAvailable ? 'Uygulanamaz'
                : d.utility.recall_target_status === 'already_met' ? 'Zaten sağlandı (Seed)'
                : d.utility.recall_target_met ? 'Başarılı' : 'Başarısız';
            const pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'Uygulanamaz';
            
            const utilityRows = [['Azınlık Sınıfı', d.utility.minority_class || '-'],
             ['Azınlık Recall (Seed)', pct(d.utility.minority_recall_seed)],
             ['Azınlık Recall (Aug)', pct(d.utility.minority_recall_augmented)],
             ['Hedef: F1 > %15 Artış', f1_target_str],
             ['Hedef: Recall > %80', recall_target_str]];
            if(d.utility.protocol) {
                utilityRows.unshift(['Test Protokolü','Train / Validation / İzole Test']);
                const selectedRecipe=d.utility.selected_recipe === 'seed_only' ? 'seed_only (sentetik eklenmedi)' : d.utility.selected_recipe || '-';
                const selectionUse=d.utility.accepted_for_utility ? 'Kabul edildi' : 'Reddedildi: fayda artışı yok';
                utilityRows.splice(1,0,
                    ['Seçilen Reçete',selectedRecipe],
                    ['Utility İçin Sentetik',selectionUse],
                    ['Üretilen Aday Satır',Number(d.utility.candidate_generated_rows || d.gen_count).toLocaleString()]
                );
                utilityRows.splice(4,0,
                    ['Macro F1 (Seed)',Number(d.utility.macro_f1_seed).toFixed(4)],
                    ['Macro F1 (Aug)',Number(d.utility.macro_f1_augmented).toFixed(4)],
                    ['Azınlık Precision (Aug)',pct(d.utility.minority_precision_augmented)],
                    ['Azınlık PR-AUC (Aug)',pct(d.utility.minority_pr_auc_augmented)]
                );
                if(d.utility.dominant_feature_diagnostic) {
                    const diag=d.utility.dominant_feature_diagnostic;
                    utilityRows.splice(3,0,['Baskın Tekil Sinyal',`${diag.feature} (F1 ${Number(diag.weighted_f1).toFixed(4)})`]);
                }
            }
            utilityRows.forEach(([l,v])=>{
                const r=document.createElement('div');r.className='detail-row';
                r.innerHTML=`<span class="label">${l}</span><span class="val">${v}</span>`;
                dl.appendChild(r);
            });
            if(Array.isArray(d.utility.selection_trials) && d.utility.selection_trials.length) {
                const hdrT=document.createElement('div');hdrT.className='detail-row';
                hdrT.innerHTML='<span class="label" style="color:var(--cyan);font-weight:700">── Reçete Deneyleri ──</span><span></span>';
                dl.appendChild(hdrT);
                d.utility.selection_trials
                    .slice()
                    .sort((a,b)=>b.objective-a.objective)
                    .slice(0,4)
                    .forEach(t=>{
                        const r=document.createElement('div');r.className='detail-row';
                        r.innerHTML=`<span class="label">${t.recipe} (${Number(t.synthetic_rows).toLocaleString()} satır)</span><span class="val">Macro F1 ${t.macro_f1.toFixed(4)} | Recall ${pct(t.minority_recall)}</span>`;
                        dl.appendChild(r);
                    });
            }
        }

        $('dl-card').classList.remove('hidden');
        if(utilityAvailable) {
            log('F1: '+d.seed_f1.toFixed(4)+' → '+d.augmented_f1.toFixed(4)+' ('+(d.improvement>0?'+':'')+d.improvement.toFixed(1)+'%)','ok');
        } else if(d.quality_report) {
            log('Kalite skoru: '+d.quality_report.overall_score.toFixed(1)+'/100 ('+d.quality_report.grade+')','ok');
        }

        if(switchPanel) {
            document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
            document.querySelector('[data-panel="analysis"]').classList.add('active');
            $('panel-analysis').classList.add('active');
            $('panel-title').textContent='Analiz';
        }
        requestAnimationFrame(()=>f1Chart.update());
    }

    // === SIMULATION — Professional MATLAB-style Engine ===
    const simC=$('simCanvas'),sctx=simC.getContext('2d');
    const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
    const median=arr=>{const a=arr.filter(Number.isFinite).slice().sort((x,y)=>x-y);return a.length?a[Math.floor(a.length/2)]:0};

    // Anomaly color map & descriptions
    const ANOMALY_COLORS={spike:'#ef4444',drift:'#f59e0b',freeze:'#22d3ee',dropout:'#8b5cf6',noise:'#ec4899'};
    const ANOMALY_NAMES={spike:'Spike',drift:'Drift',freeze:'Freeze',dropout:'Dropout',noise:'Noise'};
    const ANOMALY_DESCRIPTIONS={
        spike:{title:'Spike Anomalisi',text:'Kısa süreli ani yüksek sapma. Sensör arızası veya dış etken kaynaklı anlık veri sıçraması. Araç yörüngesinde beklenmedik zıplamalar gözlemlenir. Genellikle 1-3 zaman adımı sürer ve referans değerden ±%200\'e kadar sapma gösterebilir.'},
        drift:{title:'Drift Anomalisi',text:'Zamanla kademeli artan sapma. Sensör kalibrasyonu bozulması veya çevresel etkenler nedeniyle verinin referans değerden sistematik olarak uzaklaşması. Başlangıçta fark edilmez, zamanla kritik seviyelere ulaşır.'},
        freeze:{title:'Freeze Anomalisi',text:'Belirli bir süre aynı pozisyonda takılı kalma. Sensör donması veya veri hattı kesintisi durumunda gözlemlenir. Veri sabit bir değerde kalır, gerçek hareketi yansıtmaz. Araç hareket etmesine rağmen veri güncellenmez.'},
        dropout:{title:'Dropout Anomalisi',text:'Bazı veri noktalarının kaybolması veya kopuk iletilmesi. Ağ kesintisi, paket kaybı veya sensör arızası durumlarında gözlemlenir. Veri akışında boşluklar oluşur ve yörünge tahminini zorlaştırır.'},
        noise:{title:'Noise Anomalisi',text:'Küçük ama sürekli rastgele dalgalanmalar. Elektromanyetik girişim, düşük sinyal-gürültü oranı veya sensör hassasiyeti kaynaklı. Tek başına kritik değildir ama birikimli etki oluşturabilir.'}
    };

    let simState={running:false,paused:false,frame:0,data:null,ref:null,sim:null,frames:[],animId:null,timeChart:null,playbackSpeed:1.0,selectedType:'spike',lastAno:false};

    function simLog(msg, type='info') {
        const out = $('sim-log-output');
        if(!out) return;
        const d = document.createElement('div');
        d.className = `slog-row ${type}`;
        d.innerHTML = msg;
        out.appendChild(d);
        out.scrollTop = out.scrollHeight;
    }

    const btnClearSim = $('btn-clear-sim-logs');
    if(btnClearSim) {
        btnClearSim.addEventListener('click', () => {
            const out = $('sim-log-output');
            if(out) out.innerHTML = '<div class="slog-row info">&gt;&gt; Günlük temizlendi.</div>';
        });
    }

    // Canvas resize
    function simResize(){const wrap=simC.parentElement;if(!wrap)return;simC.width=wrap.clientWidth;simC.height=wrap.clientHeight}
    window.addEventListener('resize',simResize);
    // Defer initial resize until panel is visible
    const simResizeObserver=new MutationObserver(()=>{if($('panel-simulation').classList.contains('active')){simResize();initTimeChart()}});
    simResizeObserver.observe($('panel-simulation'),{attributes:true,attributeFilter:['class']});

    // --- Anomaly button selector ---
    document.querySelectorAll('.sim-anomaly-btn').forEach(btn=>{
        btn.addEventListener('click',()=>{
            document.querySelectorAll('.sim-anomaly-btn').forEach(b=>b.classList.remove('active'));
            btn.classList.add('active');
            simState.selectedType=btn.dataset.type;
            updateAnomalyDescription(btn.dataset.type);
            updateLegend(btn.dataset.type);
        });
    });

    function updateAnomalyDescription(type){
        const desc=ANOMALY_DESCRIPTIONS[type];
        $('sim-desc-title').textContent=desc.title;
        $('sim-desc-text').textContent=desc.text;
    }

    function updateLegend(type){
        const color=ANOMALY_COLORS[type];
        $('sim-legend-anomaly-dot').style.background=color;
        $('sim-legend-anomaly-label').textContent='Anomali ('+ANOMALY_NAMES[type]+')';
    }

    // --- Sliders ---
    $('sim-severity').addEventListener('input',e=>$('sim-severity-val').textContent=parseFloat(e.target.value).toFixed(2)+'x');
    $('sim-playback-speed').addEventListener('input',e=>{
        const v=parseFloat(e.target.value);
        $('sim-speed-val').textContent=v.toFixed(v%1===0?0:1)+'x';
        simState.playbackSpeed=v;
    });

    // --- Time Chart (Chart.js) ---
    function initTimeChart(){
        if(simState.timeChart)return;
        const ctx=$('simTimeChart');
        if(!ctx)return;
        if(!window.Chart)return;
        simState.timeChart=new Chart(ctx.getContext('2d'),{
            type:'line',
            data:{
                labels:[],
                datasets:[
                    {label:'Referans (Normal)',data:[],borderColor:'#10b981',backgroundColor:'rgba(16,185,129,.08)',borderWidth:2,pointRadius:0,tension:.3,fill:true},
                    {label:'Anomali',data:[],borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,.08)',borderWidth:2,pointRadius:0,tension:.3,fill:true}
                ]
            },
            options:{
                responsive:true,maintainAspectRatio:false,animation:{duration:0},
                scales:{
                    x:{display:true,grid:{color:'rgba(255,255,255,.03)'},ticks:{color:'#64748b',maxTicksLimit:10,font:{size:10}}},
                    y:{display:true,grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#64748b',font:{size:10}},title:{display:true,text:'Y Pozisyon (m)',color:'#64748b',font:{size:11}}}
                },
                plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false}}
            }
        });
    }

    function updateTimeChart(refData,simData,currentFrame){
        if(!simState.timeChart)return;
        const tc=simState.timeChart;
        const labels=refData.map((_,i)=>(i*0.1).toFixed(1)+'s');
        tc.data.labels=labels;
        tc.data.datasets[0].data=refData.map(v=>v.toFixed(3));
        tc.data.datasets[1].data=simData.map((v,i)=>i<=currentFrame?v.toFixed(3):null);
        tc.data.datasets[1].borderColor=ANOMALY_COLORS[simState.selectedType]||'#ef4444';
        tc.data.datasets[1].backgroundColor=(ANOMALY_COLORS[simState.selectedType]||'#ef4444')+'14';
        tc.update();
    }

    // --- Reference trajectory ---
    function makeReference(data){
        const n=data.x.length;
        const dx=[],dy=[],ds=[];
        for(let i=1;i<n;i++){dx.push(data.x[i]-data.x[i-1]);dy.push(data.y[i]-data.y[i-1]);ds.push(data.speed[i])}
        const mx=median(dx),my=median(dy),ms=Math.max(.1,median(ds));
        const ref={x:[],y:[],speed:[],vx:[],vy:[]};
        for(let i=0;i<n;i++){ref.x.push(data.x[0]+mx*i);ref.y.push(data.y[0]+my*i);ref.speed.push(ms);ref.vx.push(mx/.1);ref.vy.push(my/.1)}
        return ref;
    }

    function applySeverity(data,ref,severity){
        const n=data.x.length,sim={x:[],y:[],speed:[],vx:[],vy:[],type:data.type};
        for(let i=0;i<n;i++){
            sim.x.push(ref.x[i]+(data.x[i]-ref.x[i])*severity);
            sim.y.push(ref.y[i]+(data.y[i]-ref.y[i])*severity);
            sim.speed.push(clamp(ref.speed[i]+(data.speed[i]-ref.speed[i])*severity,0,80));
            sim.vx.push(clamp(ref.vx[i]+(data.vx[i]-ref.vx[i])*severity,-35,35));
            sim.vy.push(clamp(ref.vy[i]+(data.vy[i]-ref.vy[i])*severity,-35,35));
        }
        return sim;
    }

    function calcFrame(sim,ref,i){
        const sp=sim.speed[i];
        const accel=i>0?(sim.speed[i]-sim.speed[i-1])/.1:0;
        const offset=Math.hypot(sim.x[i]-ref.x[i],sim.y[i]-ref.y[i]);
        const jerk=i>1?Math.abs(((sim.speed[i]-sim.speed[i-1])-(sim.speed[i-1]-sim.speed[i-2]))/.01):0;
        const risk=clamp(offset*9+Math.abs(accel)*3+(sp<.4?18:0)+(jerk>120?10:0),0,100);
        return{sp,accel,offset,jerk,risk,ano:risk>45||offset>3.5||Math.abs(accel)>10||sp<.4};
    }

    // --- Vehicle canvas drawing ---
    function drawRoad(ro,mid){
        const w=simC.width,h=simC.height;
        // Background gradient
        const grd=sctx.createLinearGradient(0,0,0,h);
        grd.addColorStop(0,'#0b1220');grd.addColorStop(.5,'#080c14');grd.addColorStop(1,'#050812');
        sctx.fillStyle=grd;sctx.fillRect(0,0,w,h);
        // Grid lines (MATLAB style)
        sctx.strokeStyle='rgba(100,116,139,.06)';sctx.lineWidth=1;
        for(let gx=0;gx<w;gx+=40){sctx.beginPath();sctx.moveTo(gx,0);sctx.lineTo(gx,h);sctx.stroke()}
        for(let gy=0;gy<h;gy+=40){sctx.beginPath();sctx.moveTo(0,gy);sctx.lineTo(w,gy);sctx.stroke()}
        // Road surface
        sctx.fillStyle='rgba(255,255,255,.02)';sctx.fillRect(0,mid-132,w,264);
        sctx.strokeStyle='rgba(148,163,184,.16)';sctx.lineWidth=2;
        [mid-132,mid+132].forEach(y=>{sctx.beginPath();sctx.moveTo(0,y);sctx.lineTo(w,y);sctx.stroke()});
        // Dashed lane markers
        sctx.setLineDash([18,26]);sctx.lineWidth=1;sctx.strokeStyle='rgba(226,232,240,.18)';
        [mid-44,mid+44].forEach(y=>{sctx.beginPath();sctx.lineDashOffset=ro;sctx.moveTo(0,y);sctx.lineTo(w,y);sctx.stroke()});
        sctx.setLineDash([]);
        // Axis labels
        sctx.fillStyle='rgba(100,116,139,.4)';sctx.font='10px JetBrains Mono,monospace';
        sctx.fillText('X →',w-30,h-8);sctx.fillText('Y ↑',6,14);
    }

    function drawPath(points,color,lineWidth=2,dashed=false){
        if(points.length<2)return;
        sctx.save();sctx.strokeStyle=color;sctx.lineWidth=lineWidth;
        if(dashed)sctx.setLineDash([9,9]);
        sctx.beginPath();points.forEach((p,i)=>i?sctx.lineTo(p.x,p.y):sctx.moveTo(p.x,p.y));
        sctx.stroke();sctx.restore();
    }

    function drawCar(x,y,angle,color,ghost=false){
        sctx.save();sctx.translate(x,y);sctx.rotate(angle);
        sctx.globalAlpha=ghost?.45:1;
        sctx.shadowBlur=ghost?0:22;sctx.shadowColor=color;
        sctx.fillStyle=color;sctx.beginPath();sctx.roundRect(-24,-12,48,24,6);sctx.fill();
        sctx.fillStyle=ghost?'#101827':'#060914';sctx.fillRect(5,-9,11,18);
        sctx.fillStyle='#fbbf24';sctx.fillRect(21,-10,3,5);sctx.fillRect(21,5,3,5);
        sctx.restore();sctx.globalAlpha=1;sctx.shadowBlur=0;
    }

    // --- Controls ---
    $('btn-sim-play').addEventListener('click',async()=>{
        if(simState.running&&simState.paused){
            simState.paused=false;
            $('btn-sim-pause').disabled=false;
            $('btn-sim-play').innerHTML='<i class="fa-solid fa-play"></i> Çalışıyor';
            $('btn-sim-play').disabled=true;
            runFrame();
            return;
        }
        if(simState.running)return;
        const t=simState.selectedType;
        const severity=parseFloat($('sim-severity').value)||1;
        log('Simülasyon: '+t.toUpperCase()+' | şiddet '+severity.toFixed(2)+'x');
        try{
            const r=await fetch(API+'/api/simulation_sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:t})});
            if(!r.ok)throw new Error('Simülasyon verisi alınamadı');
            const data=await r.json();
            if(data.source==='demo')log('Demo veri kullanılıyor (önce RCGAN veri üretin)');
            startSimulation(data,severity);
        }catch(e){log(e.message,'err')}
    });

    $('btn-sim-pause').addEventListener('click',()=>{
        if(!simState.running)return;
        simState.paused=!simState.paused;
        $('btn-sim-pause').innerHTML=simState.paused?'<i class="fa-solid fa-play"></i>':'<i class="fa-solid fa-pause"></i>';
        $('btn-sim-play').disabled=!simState.paused;
        $('btn-sim-play').innerHTML=simState.paused?'<i class="fa-solid fa-play"></i> Devam':'<i class="fa-solid fa-play"></i> Çalışıyor';
        $('s-status').textContent=simState.paused?'DURDURULDU':'ÇALIŞIYOR';
        $('s-status').style.color=simState.paused?'var(--amber)':'var(--green)';
        if(!simState.paused)runFrame();
    });

    $('btn-sim-reset').addEventListener('click',()=>{
        if(simState.animId)cancelAnimationFrame(simState.animId);
        simState.running=false;simState.paused=false;simState.frame=0;simState.frames=[];
        $('btn-sim-play').disabled=false;$('btn-sim-play').innerHTML='<i class="fa-solid fa-play"></i> Başlat';
        $('btn-sim-pause').disabled=true;$('btn-sim-reset').disabled=true;
        $('btn-sim-pause').innerHTML='<i class="fa-solid fa-pause"></i>';
        $('s-status').textContent='Bekliyor';$('s-status').style.color='var(--text2)';
        $('s-speed').textContent='0.0';$('s-offset').textContent='0.0';$('s-risk').textContent='0';
        $('sim-timeline-slider').value=0;$('sim-timeline-slider').disabled=true;
        $('sim-time-display').textContent='0.0s / 0.0s';
        // Clear KPIs
        ['sk-type','sk-severity','sk-affected','sk-avg-offset','sk-max-offset','sk-loss','sk-duration','sk-risk'].forEach(id=>$(id).textContent='—');
        // Clear chart
        if(simState.timeChart){simState.timeChart.data.labels=[];simState.timeChart.data.datasets[0].data=[];simState.timeChart.data.datasets[1].data=[];simState.timeChart.update()}
        // Clear canvas
        if(simC.width>0&&simC.height>0){sctx.clearRect(0,0,simC.width,simC.height);drawRoad(0,simC.height/2)}
        const out = $('sim-log-output');
        if(out) out.innerHTML = '<div class="slog-row info">&gt;&gt; Simülasyon sıfırlandı. Yeni analiz bekleniyor...</div>';
        log('Simülasyon sıfırlandı.');
    });

    // --- Timeline scrubber ---
    $('sim-timeline-slider').addEventListener('input',e=>{
        if(!simState.sim||!simState.ref)return;
        const f=parseInt(e.target.value);
        simState.frame=f;
        renderFrameAt(f);
    });

    // --- Main simulation lifecycle ---
    function startSimulation(data,severity){
        simResize();initTimeChart();
        simState.running=true;simState.paused=false;simState.frame=0;simState.frames=[];
        simState.data=data;
        simState.ref=makeReference(data);
        simState.sim=applySeverity(data,simState.ref,severity);
        const tot=simState.sim.x.length;
        simState.lastAno=false;

        const out = $('sim-log-output');
        if(out) out.innerHTML = '';
        simLog(`&gt;&gt; [0.0s] 🚗 Simülasyon başlatıldı. Anomali Modeli: <b>${ANOMALY_NAMES[data.type]}</b> | Şiddet: <b>${severity.toFixed(2)}x</b>`, 'info');
        simLog(`&gt;&gt; [0.0s] 📊 Kaynak Veri: <b>${data.source==='rcgan'?'RCGAN Sentetik Üretici':'Deterministik Simülasyon Jeneratörü'}</b>`, 'success');
        simLog(`&gt;&gt; [0.0s] ⏱️ Toplam Zaman Aralığı: <b>${((tot-1)*0.1).toFixed(1)} sn</b> (${tot} adım)`, 'info');

        // Setup UI
        $('btn-sim-play').disabled=true;$('btn-sim-play').innerHTML='<i class="fa-solid fa-play"></i> Çalışıyor';
        $('btn-sim-pause').disabled=false;$('btn-sim-reset').disabled=false;
        $('sim-timeline-slider').max=tot-1;$('sim-timeline-slider').value=0;
        $('sim-timeline-end').textContent=((tot-1)*0.1).toFixed(1)+'s';

        // Setup time chart color
        const ac=ANOMALY_COLORS[data.type]||'#ef4444';
        if(simState.timeChart){
            simState.timeChart.data.datasets[1].borderColor=ac;
            simState.timeChart.data.datasets[1].backgroundColor=ac+'14';
        }

        // Pre-compute all frames for scrubbing
        for(let i=0;i<tot;i++){simState.frames.push(calcFrame(simState.sim,simState.ref,i))}

        // Initialize full chart data
        const refY=simState.ref.y.slice();
        const simY=simState.sim.y.slice();
        if(simState.timeChart){
            const labels=refY.map((_,i)=>(i*0.1).toFixed(1)+'s');
            simState.timeChart.data.labels=labels;
            simState.timeChart.data.datasets[0].data=refY.map(v=>parseFloat(v.toFixed(3)));
            simState.timeChart.data.datasets[1].data=simY.map(()=>null);
            simState.timeChart.update();
        }

        runFrame();
    }

    function runFrame(){
        if(!simState.running||simState.paused)return;
        const tot=simState.sim.x.length;
        if(simState.frame>=tot){
            finishSimulation();
            return;
        }

        renderFrameAt(simState.frame);

        // Reveal chart point
        if(simState.timeChart){
            simState.timeChart.data.datasets[1].data[simState.frame]=parseFloat(simState.sim.y[simState.frame].toFixed(3));
            simState.timeChart.update();
        }

        // Simülasyon Olay Günlüğü Akışı
        const m = simState.frames[simState.frame];
        if (m) {
            const timeStr = (simState.frame * 0.1).toFixed(1) + 's';
            
            // Anomali başlangıç/bitiş logları
            if (m.ano && !simState.lastAno) {
                simLog(`⚠️ <b>[${timeStr}] ANOMALİ TESPİT EDİLDİ:</b> Sapma: <b>${m.offset.toFixed(2)}m</b> | Risk: <b>%${Math.round(m.risk)}</b> (Sürüş Güvenliği Kaybı)`, 'danger');
            } else if (!m.ano && simState.lastAno) {
                simLog(`✅ <b>[${timeStr}] DURUM NORMALLEŞTİ:</b> Araç kararlı yörüngeye döndü (Sapma: ${m.offset.toFixed(2)}m)`, 'success');
            }
            
            // Periyodik durum güncellemesi (her 1.0 saniyede)
            if (simState.frame % 10 === 0 && simState.frame > 0) {
                const logType = m.ano ? 'warn' : 'info';
                simLog(`📊 <b>[${timeStr}] METRİKLER:</b> Hız: <b>${m.sp.toFixed(1)} m/s</b> | Sapma: <b>${m.offset.toFixed(2)}m</b> | Risk: <b>%${Math.round(m.risk)}</b>`, logType);
            }
            
            simState.lastAno = m.ano;
        }

        simState.frame++;
        const delay=Math.max(10,Math.round(70/simState.playbackSpeed));
        simState.animId=setTimeout(()=>requestAnimationFrame(runFrame),delay);
    }

    function renderFrameAt(f){
        const sim=simState.sim,ref=simState.ref,m=simState.frames[f];
        if(!sim||!ref||!m)return;
        const tot=sim.x.length,mid=simC.height/2,w=simC.width;
        const ac=ANOMALY_COLORS[sim.type]||'#ef4444';
        const ro=-Math.max(2,m.sp*.42)*f;

        drawRoad(ro,mid);

        // Anomaly flash
        if(m.ano){sctx.fillStyle=ac+'10';sctx.fillRect(0,0,w,simC.height)}

        // Compute visible trail
        const scale=24,step=Math.max(6,Math.min(24,w/(tot+10)));
        const sx=w*.12;
        const trailLen=Math.min(f+1,25);
        const refTrail=[],trail=[];
        for(let i=Math.max(0,f-trailLen+1);i<=f;i++){
            refTrail.push({x:sx+i*step,y:mid+(ref.y[i]-ref.y[0])*scale});
            trail.push({x:sx+i*step,y:mid+(sim.y[i]-ref.y[0])*scale});
        }

        // Draw trails
        drawPath(refTrail,'rgba(16,185,129,.45)',2,true);
        drawPath(trail,ac+'cc',3,false);

        // Trail dots
        for(let i=0;i<trail.length;i+=3){
            sctx.fillStyle=ac+Math.round(35+i*9).toString(16).padStart(2,'0');
            sctx.beginPath();sctx.arc(trail[i].x,trail[i].y,3,0,Math.PI*2);sctx.fill();
        }

        // Cars
        const refPt=refTrail[refTrail.length-1],simPt=trail[trail.length-1];
        const ag=Math.atan2(sim.vy[f],sim.vx[f]+1e-8),rag=Math.atan2(ref.vy[f],ref.vx[f]+1e-8);
        drawCar(refPt.x,refPt.y,rag,'#10b981',true);
        drawCar(simPt.x,simPt.y,ag,ac,false);

        // Deviation line
        sctx.strokeStyle='rgba(255,255,255,.18)';sctx.lineWidth=1;sctx.setLineDash([4,5]);
        sctx.beginPath();sctx.moveTo(refPt.x,refPt.y);sctx.lineTo(simPt.x,simPt.y);sctx.stroke();sctx.setLineDash([]);

        // Frame counter
        sctx.fillStyle='rgba(100,116,139,.5)';sctx.font='10px JetBrains Mono,monospace';
        sctx.fillText('Frame '+f+'/'+tot+' | '+(f*0.1).toFixed(1)+'s',8,simC.height-8);

        // Update HUD
        $('s-status').textContent=m.ano?'ANOMALİ':'STABİL';
        $('s-status').style.color=m.ano?ac:'var(--green)';
        $('s-speed').textContent=m.sp.toFixed(1);
        $('s-offset').textContent=m.offset.toFixed(2);
        $('s-risk').textContent=Math.round(m.risk);
        $('s-risk').style.color=m.risk>70?'var(--red)':m.risk>40?'var(--amber)':'var(--green)';

        // Timeline slider
        $('sim-timeline-slider').value=f;
        $('sim-time-display').textContent=(f*0.1).toFixed(1)+'s / '+((tot-1)*0.1).toFixed(1)+'s';
    }

    function finishSimulation(){
        simState.running=false;
        $('btn-sim-play').disabled=false;$('btn-sim-play').innerHTML='<i class="fa-solid fa-play"></i> Başlat';
        $('btn-sim-pause').disabled=true;
        $('sim-timeline-slider').disabled=false;
        $('s-status').textContent='BİTTİ';$('s-status').style.color='var(--green)';

        // Reveal all chart data
        if(simState.timeChart&&simState.sim){
            simState.timeChart.data.datasets[1].data=simState.sim.y.map(v=>parseFloat(v.toFixed(3)));
            simState.timeChart.update();
        }

        // Compute KPIs
        const frames=simState.frames,tot=frames.length;
        const offsets=frames.map(m=>m.offset);
        const maxOffset=Math.max(...offsets,0);
        const avgOffset=offsets.reduce((a,b)=>a+b,0)/tot;
        const affected=frames.filter(m=>m.ano).length;
        const anoTime=affected*0.1;
        const avgRisk=frames.reduce((a,m)=>a+m.risk,0)/tot;
        const maxRisk=Math.max(...frames.map(m=>m.risk),0);
        const affectedRatio=tot>0?(affected/tot):0;
        const lossRate=simState.sim.type==='dropout'?(affected/tot*100).toFixed(1)+'%':'0.0%';
        const severity=parseFloat($('sim-severity').value);
        
        let riskLevel='DÜŞÜK';
        let isDowngraded=false;
        if (maxRisk > 60) {
            // Eğer tepe risk skoru kritik seviyede ama süresi kısaysa (2.5 saniyeden kısa veya simülasyonun %30'undan azsa) ORTA risk yap
            if (anoTime < 2.5 || affectedRatio < 0.3) {
                riskLevel = 'ORTA';
                isDowngraded = true;
            } else {
                riskLevel = 'KRİTİK';
            }
        } else if (maxRisk > 35 || avgRisk > 20) {
            riskLevel = 'ORTA';
        }

        // Update KPI cards
        $('sk-type').textContent=ANOMALY_NAMES[simState.sim.type]||'—';
        $('sk-severity').textContent=severity.toFixed(2)+'x';
        $('sk-affected').textContent=affected+' / '+tot;
        $('sk-avg-offset').textContent=avgOffset.toFixed(2)+' m';
        $('sk-max-offset').textContent=maxOffset.toFixed(2)+' m';
        $('sk-loss').textContent=lossRate;
        $('sk-duration').textContent=(tot*0.1).toFixed(1)+' sn';
        $('sk-risk').textContent=riskLevel;
        $('sk-risk').style.color=riskLevel==='KRİTİK'?'var(--red)':riskLevel==='ORTA'?'var(--amber)':'var(--green)';

        // MATLAB-Style final report logging
        simLog(`------------------------------------------------------------`, 'info');
        simLog(`🏁 <b>ANALİZ TAMAMLANDI (MATLAB-Style Summary)</b>`, 'header');
        simLog(`* Anomali Modeli: <b>${ANOMALY_NAMES[simState.sim.type] || 'Bilinmiyor'}</b>`, 'info');
        simLog(`* Simülasyon Süresi: <b>${(tot*0.1).toFixed(1)} sn</b> (Etkilenen: <b>${(affected*0.1).toFixed(1)} sn</b>)`, 'info');
        simLog(`* Ortalama Sapma: <b>${avgOffset.toFixed(3)} m</b> | Maks. Sapma: <b>${maxOffset.toFixed(3)} m</b>`, 'info');
        simLog(`* Maksimum (Tepe) Risk Skoru: <b>%${Math.round(maxRisk)}</b>`, maxRisk > 60 ? 'danger' : maxRisk > 35 ? 'warn' : 'info');
        simLog(`* Ortalama Risk Skoru: <b>%${Math.round(avgRisk)}</b>`, avgRisk > 35 ? 'warn' : 'info');
        
        if (isDowngraded) {
            simLog(`* <i>Bilgi: Tepe risk kritik seviyede (%${Math.round(maxRisk)}) olsa da, anomali süresi kısa olduğu için (${anoTime.toFixed(1)} sn) genel seviye <b>ORTA RİSK</b> olarak güncellenmiştir.</i>`, 'warn');
        }

        let decisionStyle = 'color: var(--green);';
        let decisionClass = 'success';
        if (riskLevel === 'KRİTİK') {
            decisionStyle = 'color: var(--red); font-weight: 800; text-decoration: underline;';
            decisionClass = 'danger';
        } else if (riskLevel === 'ORTA') {
            decisionStyle = 'color: var(--amber); font-weight: 700;';
            decisionClass = 'warn';
        }
        simLog(`* Risk Kararı Sonucu: <span style="${decisionStyle}">${riskLevel} RİSK</span>`, decisionClass);
        simLog(`------------------------------------------------------------`, 'info');

        log('Sim tamamlandı: '+ANOMALY_NAMES[simState.sim.type]+' | risk='+riskLevel+' (maks='+Math.round(maxRisk)+', ort='+Math.round(avgRisk)+'), maks sapma='+maxOffset.toFixed(2)+' m, etkilenen='+affected+'/'+tot,'ok');
    }

    // === AUTOMATION ===
    const btnAuto = $('btn-auto');
    if(btnAuto) btnAuto.addEventListener('click',async()=>{
        const btn=$('btn-auto');btn.disabled=true;btn.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> İşleniyor...';
        $('auto-progress').classList.remove('hidden');
        const nSamples=parseInt($('n-samples').value)||2000;
        log('OTOMASYON: '+nSamples+' örnek üretiliyor...','ok');
        let p=0;
        const iv=setInterval(()=>{p+=(100-p)*.05;$('auto-thumb').style.width=p+'%';
            if(p>25&&p<30)$('auto-text').textContent='Damıtma...';
            if(p>45&&p<50)$('auto-text').textContent='Sentez...';
            if(p>70&&p<75)$('auto-text').textContent='Model eğitimi...';
        },800);
        try{
            const r=await fetch(API+'/api/run_full_automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n_samples:nSamples})});
            if(!r.ok)throw new Error('Hata:'+r.status);
            const d=await r.json();clearInterval(iv);
            $('auto-thumb').style.width='100%';$('auto-text').textContent='BAŞARILI: '+d.gen_count.toLocaleString()+' örnek!';
            btn.innerHTML='<i class="fa-solid fa-check"></i> Tamamlandı';
            showResults(d);
            log('Otomasyon tamamlandı! '+d.gen_count+' örnek','ok');
        }catch(e){clearInterval(iv);log(e.message,'err');btn.disabled=false;btn.innerHTML='<i class="fa-solid fa-bolt"></i> TÜM SÜRECİ BAŞLAT'}
    });
});
