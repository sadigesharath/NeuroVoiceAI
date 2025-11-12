let mediaRecorder;
let audioChunks = [];
let recordingStartTime;
let timerInterval;
let audioContext;
let analyser;
let dataArray;
let animationId;
let currentLanguage = 'en';
let recordedBlob = null;
let uploadedFile = null;
let pdfFilename = null;
let audioStream = null;
let recordingAudioContext = null;
let mediaStreamSource = null;
let scriptProcessor = null;
let recordedBuffers = [];

function acceptPrivacy() {
    document.getElementById('privacyNotice').style.display = 'none';
    document.getElementById('mainContent').style.display = 'block';
}

function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    
    const icon = document.querySelector('.theme-icon');
    icon.textContent = newTheme === 'dark' ? '☀️' : '🌙';
}

document.getElementById('themeToggle').addEventListener('click', toggleTheme);

function updateLanguage(lang) {
    currentLanguage = lang;
    const elements = document.querySelectorAll('[data-en]');
    
    elements.forEach(el => {
        const text = el.getAttribute(`data-${lang}`);
        if (text) {
            if (el.tagName === 'INPUT' || el.tagName === 'BUTTON') {
                if (el.hasAttribute('placeholder')) {
                    el.placeholder = text;
                } else {
                    el.textContent = text;
                }
            } else if (el.tagName === 'OPTION') {
                el.textContent = text;
            } else {
                el.textContent = text;
            }
        }
    });
}

document.getElementById('languageSelect').addEventListener('change', (e) => {
    updateLanguage(e.target.value);
});

function showRecordingInterface() {
    document.getElementById('recordingCard').style.display = 'block';
    document.getElementById('uploadCard').style.display = 'none';
    document.getElementById('recordingCard').scrollIntoView({ behavior: 'smooth' });
}

function showUploadInterface() {
    document.getElementById('uploadCard').style.display = 'block';
    document.getElementById('recordingCard').style.display = 'none';
    document.getElementById('uploadCard').scrollIntoView({ behavior: 'smooth' });
}

async function toggleRecording() {
    const recordBtn = document.getElementById('recordBtn');
    
    if (!recordingAudioContext || recordingAudioContext.state === 'closed') {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 44100
                } 
            });
            
            recordingAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
            mediaStreamSource = recordingAudioContext.createMediaStreamSource(audioStream);
            
            const bufferSize = 4096;
            scriptProcessor = recordingAudioContext.createScriptProcessor(bufferSize, 1, 1);
            recordedBuffers = [];
            
            scriptProcessor.onaudioprocess = (e) => {
                const channelData = e.inputBuffer.getChannelData(0);
                recordedBuffers.push(new Float32Array(channelData));
            };
            
            mediaStreamSource.connect(scriptProcessor);
            scriptProcessor.connect(recordingAudioContext.destination);
            
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<span class="record-icon">⏹</span><span>Stop Recording</span>';
            
            recordingStartTime = Date.now();
            updateTimer();
            timerInterval = setInterval(updateTimer, 1000);
            
            setupVisualizer(audioStream);
            
        } catch (error) {
            alert('Error accessing microphone: ' + error.message);
        }
    } else {
        if (scriptProcessor) {
            scriptProcessor.disconnect();
        }
        if (mediaStreamSource) {
            mediaStreamSource.disconnect();
        }
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
        }
        
        recordBtn.classList.remove('recording');
        recordBtn.innerHTML = '<span class="record-icon">●</span><span>Start Recording</span>';
        clearInterval(timerInterval);
        cancelAnimationFrame(animationId);
        
        const wavBlob = exportWAV(recordedBuffers, recordingAudioContext.sampleRate);
        recordedBlob = wavBlob;
        
        const audioUrl = URL.createObjectURL(wavBlob);
        const audioPlayback = document.getElementById('audioPlayback');
        audioPlayback.src = audioUrl;
        audioPlayback.load();
        audioPlayback.style.display = 'block';
        document.getElementById('analyzeRecordingBtn').style.display = 'block';
        
        console.log('Recording complete. Blob size:', wavBlob.size, 'bytes');
        
        if (recordingAudioContext) {
            recordingAudioContext.close();
            recordingAudioContext = null;
        }
    }
}

function exportWAV(buffers, sampleRate) {
    const totalLength = buffers.reduce((acc, buffer) => acc + buffer.length, 0);
    const mergedBuffer = new Float32Array(totalLength);
    
    let offset = 0;
    for (const buffer of buffers) {
        mergedBuffer.set(buffer, offset);
        offset += buffer.length;
    }
    
    const pcmData = floatTo16BitPCM(mergedBuffer);
    const wavBuffer = createWAVFile(pcmData, sampleRate);
    
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function floatTo16BitPCM(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
}

function createWAVFile(pcmData, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcmData.length * 2;
    const headerSize = 44;
    
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buffer);
    
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    let offset = 44;
    for (let i = 0; i < pcmData.length; i++) {
        view.setInt16(offset, pcmData[i], true);
        offset += 2;
    }
    
    return buffer;
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    document.getElementById('timer').textContent = `${minutes}:${seconds}`;
}

function setupVisualizer(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
    
    visualize();
}

function visualize() {
    const canvas = document.getElementById('visualizer');
    const canvasCtx = canvas.getContext('2d');
    
    animationId = requestAnimationFrame(visualize);
    
    analyser.getByteFrequencyData(dataArray);
    
    canvasCtx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--background');
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    const barWidth = (canvas.width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height;
        
        const gradient = canvasCtx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, '#3f51b5');
        gradient.addColorStop(1, '#ff4081');
        canvasCtx.fillStyle = gradient;
        
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('selectedFile').style.display = 'block';
    }
}

const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    
    const file = e.dataTransfer.files[0];
    if (file) {
        uploadedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('selectedFile').style.display = 'block';
    }
});

async function analyzeRecording() {
    if (!recordedBlob) {
        alert('Please record audio first');
        return;
    }
    
    const form = document.getElementById('patientForm');
    if (!form.checkValidity()) {
        alert('Please fill in all patient information fields');
        form.reportValidity();
        return;
    }
    
    const formData = new FormData();
    formData.append('audio', recordedBlob, 'recording.wav');
    formData.append('name', document.getElementById('name').value);
    formData.append('age', document.getElementById('age').value);
    formData.append('gender', document.getElementById('gender').value);
    
    await performAnalysis(formData);
}

async function analyzeUploadedFile() {
    if (!uploadedFile) {
        alert('Please select a file first');
        return;
    }
    
    const form = document.getElementById('patientForm');
    if (!form.checkValidity()) {
        alert('Please fill in all patient information fields');
        form.reportValidity();
        return;
    }
    
    const formData = new FormData();
    formData.append('audio', uploadedFile);
    formData.append('name', document.getElementById('name').value);
    formData.append('age', document.getElementById('age').value);
    formData.append('gender', document.getElementById('gender').value);
    
    await performAnalysis(formData);
}

// ... [rest of your code above is unchanged] ...

async function performAnalysis(formData) {
    document.getElementById('loadingOverlay').style.display = 'flex';

    try {
        const response = await fetch('http://localhost:5000/analyze', { // <-- FIXED LINE!
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);

    } catch (error) {
        alert('Analysis failed: ' + error.message);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
}

// ... [rest of your code below is unchanged] ...
function displayResults(data) {
    const resultsCard = document.getElementById('resultsCard');
    const resultSummary = document.getElementById('resultSummary');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    
    pdfFilename = data.pdf_filename;
    
    if (data.prediction === 0) {
        resultSummary.className = 'result-summary healthy';
        resultIcon.textContent = '✅';
        resultText.textContent = currentLanguage === 'en' ? 'Healthy Voice Pattern' : 'स्वस्थ आवाज पैटर्न';
    } else {
        resultSummary.className = 'result-summary parkinsons';
        resultIcon.textContent = '⚠️';
        resultText.textContent = currentLanguage === 'en' ? "Parkinson's Disease Detected" : "पार्किंसंस रोग का पता चला";
    }
    
    const confidencePercent = (data.confidence * 100).toFixed(1);
    confidenceFill.style.width = confidencePercent + '%';
    confidenceValue.textContent = confidencePercent + '%';
    
    const featuresList = document.getElementById('featuresList');
    featuresList.innerHTML = '';
    
    const featureNames = {
        'jitter': currentLanguage === 'en' ? 'Jitter (Voice Instability)' : 'जिटर (आवाज अस्थिरता)',
        'shimmer': currentLanguage === 'en' ? 'Shimmer (Amplitude Variation)' : 'शिमर (आयाम भिन्नता)',
        'hnr': currentLanguage === 'en' ? 'Harmonic-to-Noise Ratio' : 'हार्मोनिक-टू-नॉइज़ रेशियो',
        'mfcc_mean': currentLanguage === 'en' ? 'MFCC Mean (Voice Quality)' : 'एमएफसीसी मीन',
        'mfcc_std': currentLanguage === 'en' ? 'MFCC Variation' : 'एमएफसीसी भिन्नता',
        'pitch_mean': currentLanguage === 'en' ? 'Average Pitch' : 'औसत पिच',
        'pitch_std': currentLanguage === 'en' ? 'Pitch Variation' : 'पिच भिन्नता',
        'energy_mean': currentLanguage === 'en' ? 'Voice Energy' : 'आवाज ऊर्जा',
        'spectral_centroid': currentLanguage === 'en' ? 'Spectral Centroid' : 'स्पेक्ट्रल सेंट्रॉइड',
        'zero_crossing_rate': currentLanguage === 'en' ? 'Zero Crossing Rate' : 'जीरो क्रॉसिंग रेट'
    };
    
    data.top_features.forEach(feature => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        featureItem.innerHTML = `
            <div class="feature-name">${featureNames[feature.name] || feature.name}</div>
            <div class="feature-value">${currentLanguage === 'en' ? 'Value' : 'मान'}: ${feature.value.toFixed(4)} | ${currentLanguage === 'en' ? 'Importance' : 'महत्व'}: ${(feature.importance * 100).toFixed(1)}%</div>
        `;
        featuresList.appendChild(featureItem);
    });
    
    const recommendationText = document.getElementById('recommendationText');
    if (data.prediction === 1) {
        recommendationText.innerHTML = currentLanguage === 'en' 
            ? `
                <p><strong>⚕️ Please consult a neurologist for professional evaluation.</strong></p>
                <p>Early detection and intervention can significantly improve quality of life. Consider scheduling an appointment with a movement disorder specialist.</p>
                <p><strong>Helpful Resources:</strong></p>
                <ul>
                    <li>Contact your primary care physician for referral</li>
                    <li>Parkinson's Foundation: <a href="https://www.parkinson.org" target="_blank">parkinson.org</a></li>
                    <li>Michael J. Fox Foundation: <a href="https://www.michaeljfox.org" target="_blank">michaeljfox.org</a></li>
                </ul>
            `
            : `
                <p><strong>⚕️ कृपया पेशेवर मूल्यांकन के लिए न्यूरोलॉजिस्ट से परामर्श करें।</strong></p>
                <p>प्रारंभिक पहचान और हस्तक्षेप जीवन की गुणवत्ता में उल्लेखनीय सुधार कर सकता है।</p>
            `;
    } else {
        recommendationText.innerHTML = currentLanguage === 'en'
            ? `
                <p><strong>✓ Your voice analysis shows healthy patterns.</strong></p>
                <p>However, if you experience any symptoms such as tremors, stiffness, or movement difficulties, please consult a healthcare professional.</p>
                <p>Regular health checkups are recommended for preventive care.</p>
            `
            : `
                <p><strong>✓ आपकी आवाज विश्लेषण स्वस्थ पैटर्न दिखाता है।</strong></p>
                <p>हालांकि, यदि आप कोई लक्षण अनुभव करते हैं, तो कृपया स्वास्थ्य पेशेवर से परामर्श करें।</p>
            `;
    }
    
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

function downloadReport() {
    if (pdfFilename) {
        window.location.href = `/download/${pdfFilename}`;
    }
}

function resetTest() {
    recordedBlob = null;
    uploadedFile = null;
    pdfFilename = null;
    recordedBuffers = [];
    
    if (recordingAudioContext && recordingAudioContext.state !== 'closed') {
        recordingAudioContext.close();
    }
    recordingAudioContext = null;
    
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
    
    document.getElementById('patientForm').reset();
    document.getElementById('recordingCard').style.display = 'none';
    document.getElementById('uploadCard').style.display = 'none';
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('audioPlayback').style.display = 'none';
    document.getElementById('analyzeRecordingBtn').style.display = 'none';
    document.getElementById('selectedFile').style.display = 'none';
    document.getElementById('timer').textContent = '00:00';
    
    const recordBtn = document.getElementById('recordBtn');
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = '<span class="record-icon">●</span><span>Start Recording</span>';
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
