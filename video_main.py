<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text-to-Image Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f6f8;
            margin: 0;
            padding: 0;
            text-align: center;
        }
        h1 {
            margin-top: 30px;
            font-size: 1.8rem;
            color: #222;
        }
        .container {
            max-width: 700px;
            margin: auto;
            background: #fff;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        input, button {
            padding: 10px;
            font-size: 1rem;
            border-radius: 6px;
            border: 1px solid #ccc;
            outline: none;
            margin: 5px 0;
        }
        input {
            width: 80%;
            max-width: 400px;
        }
        button {
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        #logs {
            margin-top: 15px;
            padding: 10px;
            max-height: 250px;
            overflow-y: auto;
            background-color: #f9f9f9;
            border-radius: 8px;
            text-align: left;
            font-family: monospace;
            white-space: pre-line;
        }
        #result img {
            margin-top: 15px;
            border-radius: 8px;
            max-width: 100%;
        }
        .download-btn {
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background-color: #28a745;
            color: white;
            border-radius: 6px;
            text-decoration: none;
        }
        .download-btn:hover {
            background-color: #1e7e34;
        }
    </style>
</head>
<body>

<h1>AI Text-to-Image Generator</h1>

<div class="container">
    <p>Enter a prompt and let the AI generate an image. You can download the result after generation.</p>

    <input type="text" id="prompt" placeholder="Describe your image...">
    <br>
    <button onclick="generateImage()">Generate Image</button>

    <div id="logs"></div>
    <div id="result"></div>
</div>

<script>
    function addLog(msg) {
        const logsEl = document.getElementById("logs");
        logsEl.innerHTML += msg + "\n";
        logsEl.scrollTop = logsEl.scrollHeight;
        console.log(msg); // also log to browser console
    }

    async function generateImage() {
        const prompt = document.getElementById("prompt").value.trim();
        const logsEl = document.getElementById("logs");
        const resultEl = document.getElementById("result");
        logsEl.innerHTML = "";
        resultEl.innerHTML = "";

        if (!prompt) {
            alert("Please enter a prompt!");
            return;
        }

        addLog("Sending prompt to server...");

        try {
            const response = await fetch("https://ai-image-backend-sj2c.onrender.com/generate-image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt })
            });

            const data = await response.json();

            if (data.logs) {
                data.logs.forEach(log => addLog(log));
            }

            if (data.error) {
                addLog("Error: " + data.error);
                return;
            }

            if (data.image_url) {
                addLog("Image generated successfully.");
                resultEl.innerHTML = `<img src="${data.image_url}" alt="Generated Image">`;
            }

            if (data.download_url) {
                const downloadBtn = document.createElement("a");
                downloadBtn.href = `https://ai-image-backend-sj2c.onrender.com${data.download_url}`;
                downloadBtn.download = "generated_image.png";
                downloadBtn.className = "download-btn";
                downloadBtn.textContent = "Download Image";
                resultEl.appendChild(downloadBtn);
            }

        } catch (err) {
            addLog("Error: " + err.message);
        }
    }
</script>

</body>
</html>
