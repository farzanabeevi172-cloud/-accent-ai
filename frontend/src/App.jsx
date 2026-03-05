import React, { useEffect, useRef, useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000/predict";

const App = () => {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [noVoice, setNoVoice] = useState(false);

  const animationRef = useRef(null);

  useEffect(() => {
    return () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
    };
  }, [mediaRecorder]);

  const startRecording = async () => {
    try {
      setError(null);
      setResult(null);
      setNoVoice(false);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const options = { mimeType: "audio/webm" };
      const recorder = new MediaRecorder(stream, options);

      const localChunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          localChunks.push(e.data);
        }
      };

      recorder.onstop = async () => {
        setLoading(true);
        try {
          const blob = new Blob(localChunks, { type: "audio/webm" });
          await sendToApi(blob);
        } catch (e) {
          setError("Failed to process audio. Please try again.");
        } finally {
          setLoading(false);
        }
      };

      setChunks(localChunks);
      setMediaRecorder(recorder);
      recorder.start();
      setRecording(true);
    } catch (e) {
      console.error(e);
      setError("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      setRecording(false);
    }
  };

  const sendToApi = async (blob) => {
    const formData = new FormData();
    // Backend expects a WAV file, but librosa can decode various containers if ffmpeg is installed.
    // We still name it .wav for compatibility.
    formData.append("file", blob, "recording.wav");

    try {
      const response = await axios.post(API_URL, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const data = response.data;
      if (data.status === "no_voice") {
        setNoVoice(true);
        setResult(null);
      } else if (data.accent && typeof data.confidence === "number") {
        setResult({
          accent: data.accent,
          confidence: (data.confidence * 100).toFixed(1),
        });
        setNoVoice(false);
      } else {
        setError("Unexpected response from server.");
      }
    } catch (err) {
      console.error(err);
      setError("Error calling backend. Make sure the API is running.");
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <h1 className="title">Accent AI Detector</h1>
        <p className="subtitle">
          Record a short 3–5 second sample of English speech and we&apos;ll
          detect whether it sounds Indian, American, or British.
        </p>

        <div className="record-section">
          <button
            className={`record-button ${recording ? "recording" : ""}`}
            onClick={recording ? stopRecording : startRecording}
          >
            {recording ? "Stop Recording" : "Start Recording"}
          </button>

          <div className="record-hint">
            {recording ? "Recording... speak clearly into the microphone." : "Click to record. Speak for at least 3 seconds."}
          </div>

          {recording && (
            <div className="wave-wrapper" ref={animationRef}>
              <div className="wave-bar bar1" />
              <div className="wave-bar bar2" />
              <div className="wave-bar bar3" />
              <div className="wave-bar bar4" />
              <div className="wave-bar bar5" />
            </div>
          )}
        </div>

        {loading && <div className="info">Analyzing your accent...</div>}

        {noVoice && (
          <div className="warning">No Voice Detected. Please try again.</div>
        )}

        {result && !noVoice && (
          <div className="result-card">
            <div className="result-label">Detected Accent</div>
            <div className="result-accent">{result.accent}</div>
            <div className="result-confidence">
              Confidence: <span>{result.confidence}%</span>
            </div>
          </div>
        )}

        {error && <div className="error">{error}</div>}

        <div className="footer">
          Backend: <code>http://localhost:8000/predict</code>
        </div>
      </div>
    </div>
  );
};

export default App;

