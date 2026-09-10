#include <M5Unified.h>
#include <SD.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <arduinoFFT.h> // Required for DSP math

// --- CONFIGURATION ---
const char* ssid = "Galaxy M34 5G 56FC";         
const char* password = "a52aje8zarxkjrz"; 
const char* serverUrl = "http://10.88.0.195:8000/upload"; 

const int SAMPLE_RATE = 16000;
const int RECORD_TIME_SEC = 5;
const int VOLUME_THRESHOLD = 500; 
const int BUFFER_SIZE = 1024;

int16_t mic_buffer[BUFFER_SIZE];

// --- FFT CONFIGURATION ---
double vReal[BUFFER_SIZE];
double vImag[BUFFER_SIZE];
// Using arduinoFFT v2 syntax
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, BUFFER_SIZE, SAMPLE_RATE);

// --- FUNCTION PROTOTYPES ---
void recordAnalyzeAndUpload();
void writeWavHeader(File& file, uint32_t dataSize, uint32_t sampleRate);
void sendToServer(String filename);
String getFilename(int counter) {
    char buf[25];
    sprintf(buf, "silence_%03d.wav", counter);
    return String(buf);
}
void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    
    M5.Display.setTextSize(2);
    M5.Display.println("Deforestation Sys");
    
    // 1. WiFi Setup
    WiFi.begin(ssid, password);
    M5.Display.print("Connecting WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        M5.Display.print(".");
    }
    M5.Display.println("\nWiFi Connected!");

    // 2. SD Card Setup
    if (!SD.begin(4, SPI, 25000000)) {
        M5.Display.setTextColor(TFT_RED);
        M5.Display.println("SD Error!");
        while (1);
    }

    // 3. Microphone Setup
    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate = SAMPLE_RATE;
    M5.Mic.config(mic_cfg);
    M5.Mic.begin();
    
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.println("Monitoring Forest...");
}

void loop() {
    M5.update();
    
    // The Gatekeeper: Listen for loud volume spikes
    if (M5.Mic.record(mic_buffer, BUFFER_SIZE, SAMPLE_RATE)) {
        long total_volume = 0;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            total_volume += abs(mic_buffer[i]);
        }
        int avg_volume = total_volume / BUFFER_SIZE;
        
        if (avg_volume > VOLUME_THRESHOLD) {
            M5.Display.fillScreen(TFT_RED);
            M5.Display.setCursor(0, 0);
            M5.Display.println("🔊 ALERT TRIGGERED");
            M5.Display.printf("Vol: %d\n", avg_volume);
            
            // Execute the DSP Analysis and Upload logic
            recordAnalyzeAndUpload(); 

            M5.Display.fillScreen(TFT_BLACK);
            M5.Display.setCursor(0, 0);
            M5.Display.setTextColor(TFT_WHITE);
            M5.Display.println("Monitoring Forest...");
        }
    }
}

void recordAnalyzeAndUpload() {
    String filename = "/rec_" + String(millis()) + ".wav";
    File audioFile = SD.open(filename, FILE_WRITE);
    
    if (!audioFile) return;

    // Write placeholder for Header
    uint8_t dummyHeader[44] = {0};
    audioFile.write(dummyHeader, 44);

    int total_samples = 0;
    int target_samples = SAMPLE_RATE * RECORD_TIME_SEC;
    
    int total_frames = 0;
    int hacksaw_frames = 0;
    
    M5.Display.println("Recording & DSP...");

    while (total_samples < target_samples) {
        if (M5.Mic.record(mic_buffer, BUFFER_SIZE, SAMPLE_RATE)) {
            
            // 1. Write the raw audio chunk directly to the SD card
            audioFile.write((uint8_t*)mic_buffer, BUFFER_SIZE * 2);
            
            // 2. Prepare this exact chunk for FFT Math
            for (int i = 0; i < BUFFER_SIZE; i++) {
                vReal[i] = (double)mic_buffer[i];
                vImag[i] = 0.0;
            }
            
            // 3. Compute the Fast Fourier Transform
            FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
            FFT.compute(FFTDirection::Forward);
            FFT.complexToMagnitude();
            
            // 4. Analyze Frequencies (16000Hz / 1024 bins = 15.625Hz per bin)
            // Hacksaw target: 2000Hz to 5000Hz (roughly bins 128 to 320)
            double target_energy = 0;
            double total_energy = 0;
            
            for (int i = 2; i < (BUFFER_SIZE / 2); i++) { // Skip DC offset (bin 0 and 1)
                total_energy += vReal[i];
                if (i >= 128 && i <= 320) {
                    target_energy += vReal[i];
                }
            }
            
            // If more than 35% of the acoustic energy in this 0.06s chunk is in the saw band
            if (total_energy > 0 && (target_energy / total_energy) > 0.35) {
                hacksaw_frames++;
            }

            total_frames++;
            total_samples += BUFFER_SIZE;
        }
    }

    // Finalize the WAV file on the SD Card
    uint32_t dataSize = total_samples * 2;
    writeWavHeader(audioFile, dataSize, SAMPLE_RATE);
    audioFile.close();

    // 5. The Final Prediction Logic
    // If at least 15% of the recording contains high-energy mechanical scraping sounds
    float saw_percentage = (float)hacksaw_frames / total_frames;
    
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0, 0);
    
    if (saw_percentage > 0.15) {
        M5.Display.setTextColor(TFT_RED);
        M5.Display.printf("SAW DETECTED!\nScore: %.2f\n", saw_percentage);
        M5.Display.println("Uploading...");
        
        sendToServer(filename); // Execute your WiFi upload function
    } else {
        M5.Display.setTextColor(TFT_GREEN);
        M5.Display.printf("Nature/Wind\nScore: %.2f\n", saw_percentage);
        M5.Display.println("SD Saved. No Upload.");
        delay(3000); // Let the user read the screen before resetting
    }
}

void sendToServer(String filename) {
    if (WiFi.status() != WL_CONNECTED) return;

    File file = SD.open(filename, FILE_READ);
    if (!file) return;

    size_t fileSize    = file.size();
    uint8_t* audioData = (uint8_t*)malloc(fileSize);
    file.read(audioData, fileSize);
    file.close();

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("X-Device-Id", "DEV-002");
    http.addHeader("X-Api-Key",   "key-dev-002-secret");
---
    String boundary = "----ESP32Boundary123";
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

    String head = "--" + boundary + "\r\n"
                  "Content-Disposition: form-data; name=\"audio\"; filename=\""
                  + getFilename(filename.substring(1).toInt()) + "\"\r\n"
                  "Content-Type: audio/wav\r\n\r\n";

    String tail = "\r\n--" + boundary + "\r\n"
                  "Content-Disposition: form-data; name=\"duration\"\r\n\r\n"
                  + String(RECORD_TIME_SEC) + "\r\n"
                  "--" + boundary + "--\r\n";

    size_t totalSize = head.length() + fileSize + tail.length();
    uint8_t* body    = (uint8_t*)malloc(totalSize);
    memcpy(body,                            head.c_str(), head.length());
    memcpy(body + head.length(),            audioData,    fileSize);
    memcpy(body + head.length() + fileSize, tail.c_str(), tail.length());
    free(audioData);

    http.setTimeout(30000);
    int httpCode = http.POST(body, totalSize);
    free(body);

    if (httpCode == 200) {
        M5.Display.setTextColor(TFT_GREEN);
        M5.Display.println("Upload OK!");
    } else {
        M5.Display.setTextColor(TFT_ORANGE);
        M5.Display.printf("Upload failed: %d\n", httpCode);
    }

    M5.Display.setTextColor(TFT_WHITE);
    http.end();
}

void writeWavHeader(File& file, uint32_t dataSize, uint32_t sampleRate) {
    uint32_t fileSize = 36 + dataSize;
    uint32_t byteRate = sampleRate * 2; 
    uint8_t header[44];
    memcpy(header, "RIFF", 4);
    header[4] = fileSize & 0xFF; header[5] = (fileSize >> 8) & 0xFF;
    header[6] = (fileSize >> 16) & 0xFF; header[7] = (fileSize >> 24) & 0xFF;
    memcpy(header + 8, "WAVEfmt ", 8);
    header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
    header[20] = 1; header[21] = 0; // PCM
    header[22] = 1; header[23] = 0; // Mono
    header[24] = sampleRate & 0xFF; header[25] = (sampleRate >> 8) & 0xFF;
    header[26] = (sampleRate >> 16) & 0xFF; header[27] = (sampleRate >> 24) & 0xFF;
    header[28] = byteRate & 0xFF; header[29] = (byteRate >> 8) & 0xFF;
    header[30] = (byteRate >> 16) & 0xFF; header[31] = (byteRate >> 24) & 0xFF;
    header[32] = 2; header[33] = 0;
    header[34] = 16; header[35] = 0;
    memcpy(header + 36, "data", 4);
    header[40] = dataSize & 0xFF; header[41] = (dataSize >> 8) & 0xFF;
    header[42] = (dataSize >> 16) & 0xFF; header[43] = (dataSize >> 24) & 0xFF;
    file.seek(0);
    file.write(header, 44);
}