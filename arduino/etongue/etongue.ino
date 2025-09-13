#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define PH_PIN A0
#define TDS_PIN A1
#define ONE_WIRE_BUS 2
#define BUTTON_PIN 3

Adafruit_ADS1115 ads;

const unsigned long SAMPLING_DURATION_MS = 5000;
const unsigned long SAMPLE_INTERVAL_MS = 200;

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

bool sentReading = false;
bool adsError = false;

// --- pH calibration placeholders ---
float ph4Voltage = 3.00;
float ph7Voltage = 2.50;

void setup() {
  Serial.begin(9600);
  sensors.begin();
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (!ads.begin()) {
    adsError = true; // mark ADS failed, but don't block
  }
  ads.setGain(GAIN_ONE);
}

void loop() {
  if (!sentReading) {
    takeAndSendMeasurement();
    sentReading = true;
  } else {
    waitForNextTrigger();
  }
}

void takeAndSendMeasurement() {
  unsigned long startMs = millis();
  unsigned long sampleCount = 0;
  float phSum = 0.0, tdsSum = 0.0, tempSum = 0.0, conductivitySum = 0.0;

  while ((millis() - startMs) < SAMPLING_DURATION_MS) {
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);
    if (isnan(tempC) || tempC <= -100.0) tempC = 25.0;

    int phRaw = analogRead(PH_PIN);
    int tdsRaw = analogRead(TDS_PIN);

    float phVoltage = phRaw * (5.0 / 1023.0);
    float tdsVoltage = tdsRaw * (5.0 / 1023.0);

    float conductivity = 0;
    if (!adsError) {
      int16_t adsValue = ads.readADC_SingleEnded(0);
      float condVoltage = adsValue * (4.096 / 32767.0);
      conductivity = condVoltage * 1000.0; // adjust calibration as needed
    }

    // pH calculation
    float slope = (7.0 - 4.0) / (ph7Voltage - ph4Voltage);
    float intercept = 7.0 - slope * ph7Voltage;
    float pH = slope * phVoltage + intercept;
    if (isnan(pH)) pH = 7.0;
    if (pH < 0.0) pH = 0.0;
    if (pH > 14.0) pH = 14.0;

    // TDS calculation
    float compensation = 1.0 + 0.02 * (tempC - 25.0);
    if (compensation <= 0.0) compensation = 1.0;
    float tds = (133.42 * pow(tdsVoltage / compensation, 3)
               - 255.86 * pow(tdsVoltage / compensation, 2)
               + 857.39 * (tdsVoltage / compensation)) * 0.5;
    if (isnan(tds) || tds < 0.0) tds = 0.0;

    phSum += pH;
    tdsSum += tds;
    tempSum += tempC;
    conductivitySum += conductivity;
    sampleCount++;

    delay(SAMPLE_INTERVAL_MS);
  }

  // compute averages
  float pH_avg = phSum / sampleCount;
  float tds_avg = tdsSum / sampleCount;
  float temp_avg = tempSum / sampleCount;
  float cond_avg = conductivitySum / sampleCount;

  // send JSON
  String jsonData = "{";
  jsonData += "\"Solids_TDS\":" + String(tds_avg, 2) + ",";
  jsonData += "\"Conductivity\":" + String(cond_avg, 2) + ",";
  jsonData += "\"pH\":" + String(pH_avg, 2) + ",";
  jsonData += "\"temperature\":" + String(temp_avg, 2);
  jsonData += "}";

  Serial.println(jsonData);
}

void waitForNextTrigger() {
  while (true) {
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == 'n' || c == 'N') {
        sentReading = false;
        while (Serial.available() > 0) Serial.read();
        delay(50);
        return;
      }
    }
    if (digitalRead(BUTTON_PIN) == LOW) {
      delay(50);
      if (digitalRead(BUTTON_PIN) == LOW) {
        while (digitalRead(BUTTON_PIN) == LOW) delay(10);
        sentReading = false;
        delay(50);
        return;
      }
    }
    delay(50);
  }
}
