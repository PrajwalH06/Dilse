let mediaRecorder;
let audioChunks = [];
let stopTimeout;

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstart = () => {
            document.getElementById("status").innerText = "🔴 Recording...";
            toggleButtons(true);
            console.log("Recording started...");
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const file = new File([audioBlob], "voice.wav", { type: 'audio/wav' });

            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            document.getElementById("audio").files = dataTransfer.files;

            document.getElementById("status").innerText = "🛑 Recording stopped.";
            toggleButtons(false);
            clearTimeout(stopTimeout);
            console.log("Recording stopped. Audio ready.");
        };

        mediaRecorder.start();

        // Optional timeout - stops recording after 60 seconds
        stopTimeout = setTimeout(() => {
            if (mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                alert("Recording automatically stopped after 60 seconds.");
            }
        }, 60000);
    }).catch(error => {
        console.error("Microphone access denied or error:", error);
        alert("Microphone access denied. Please allow microphone permissions.");
    });
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    } else {
        console.warn("No active recording to stop.");
    }
}

function toggleButtons(recording) {
    document.querySelector("button[onclick='startRecording()']").disabled = recording;
    document.querySelector("button[onclick='stopRecording()']").disabled = !recording;
}
