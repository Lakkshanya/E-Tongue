#include <OneWire.h>
#include <DallasTemperature.h>

#define PH_PIN A0
#define TDS_PIN A1
#define ONE_WIRE_BUS 2
#define BUTTON_PIN 3           // optional: connect a momentary push-button to D3 (to GND), use INPUT_PULLUP

// Sampling configuration
const unsigned long SAMPLING_DURATION_MS = 5000;  // 5 seconds analysis time
const unsigned long SAMPLE_INTERVAL_MS   = 200;   // sample every 200 ms -> ~25 samples in 5s

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

bool sentReading = false; // becomes true after sending one JSON for current solution

// --- pH calibration placeholders ---
// Measure actual voltages in buffers and replace these!
float ph4Voltage = 3.00;   // MEASURE and replace with your pH4 buffer voltage
float ph7Voltage = 2.50;   // MEASURE and replace with your pH7 buffer voltage

void setup() {
  Serial.begin(9600);
  sensors.begin();
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  // no other Serial prints are used — only JSON output to avoid confusing backend parser
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
  unsigned long elapsed = 0;

  unsigned long sampleCount = 0;
  unsigned long phRawSum = 0;
  unsigned long tdsRawSum = 0;
  float tempSum = 0.0;

  // sample loop for SAMPLING_DURATION_MS
  while ((millis() - startMs) < SAMPLING_DURATION_MS) {
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    // handle disconnected temp sensor (Dallas returns -127 or DEVICE_DISCONNECTED_C)
    if (isnan(tempC) || tempC <= -100.0) {
      tempC = 25.0; // fallback
    }

    int phRaw = analogRead(PH_PIN);
    int tdsRaw = analogRead(TDS_PIN);

    phRawSum += phRaw;
    tdsRawSum += tdsRaw;
    tempSum += tempC;
    sampleCount++;

    delay(SAMPLE_INTERVAL_MS);
  }

  // compute averages
  float phRawAvg = phRawSum / (float)sampleCount;
  float tdsRawAvg = tdsRawSum / (float)sampleCount;
  float tempAvg = tempSum / (float)sampleCount;

  // convert to voltages (Arduino UNO 10-bit ADC, 5V ref)
  float phVoltage = phRawAvg * (5.0 / 1023.0);
  float tdsVoltage = tdsRawAvg * (5.0 / 1023.0);

  // --- pH mapping (linear) ---
  float slope = (7.0 - 4.0) / (ph7Voltage - ph4Voltage);
  float intercept = 7.0 - slope * ph7Voltage;
  float pH = slope * phVoltage + intercept;
  if (isnan(pH)) pH = 7.0;
  if (pH < 0.0) pH = 0.0;
  if (pH > 14.0) pH = 14.0;

  // --- TDS calculation with temperature compensation ---
  float compensationCoefficient = 1.0 + 0.02 * (tempAvg - 25.0);
  if (compensationCoefficient <= 0.0) compensationCoefficient = 1.0; // safety
  float compensatedVoltage = tdsVoltage / compensationCoefficient;
  float tds = (133.42 * pow(compensatedVoltage, 3)
             - 255.86 * pow(compensatedVoltage, 2)
             + 857.39 * compensatedVoltage) * 0.5; // keep 0.5 scaling from earlier formula

  if (isnan(tds) || tds < 0.0) tds = 0.0;

  // --- Build JSON (single line) ---
  // Keep the same fields your backend expects:
  String jsonData = "{";
  jsonData += "\"Hardness\":180,";
  jsonData += "\"Solids_TDS\":" + String(tds, 2) + ",";
  jsonData += "\"Sulphate\":320,";
  jsonData += "\"Chloramine\":6.5,";
  jsonData += "\"Conductivity\":400,";
  jsonData += "\"Organic_Carbon\":7,";
  jsonData += "\"Trihalomethane\":70,";
  jsonData += "\"Turbidity\":4,";
  jsonData += "\"pH\":" + String(pH, 2) + ",";
  jsonData += "\"temperature\":" + String(tempAvg, 2);
  jsonData += "}";

  Serial.println(jsonData);    // ONE single JSON line per solution
}

void waitForNextTrigger() {
  // Wait for either:
  //  - Serial 'n' or 'N' (recommended: send from Node backend after it finishes prediction)
  //  - OR a physical button press on D3 (active LOW)
  // This loop blocks but is simple and ensures only 1 JSON is ever sent until user triggers next.
  while (true) {
    // serial trigger
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == 'n' || c == 'N') {
        sentReading = false;
        // flush any other bytes in buffer
        while (Serial.available() > 0) Serial.read();
        delay(50); // small debounce
        return;
      }
    }

    // button trigger (active LOW, using INPUT_PULLUP)
    if (digitalRead(BUTTON_PIN) == LOW) {
      // debounce
      delay(50);
      if (digitalRead(BUTTON_PIN) == LOW) {
        // wait until released
        while (digitalRead(BUTTON_PIN) == LOW) delay(10);
        sentReading = false;
        delay(50);
        return;
      }
    }
    delay(50);
  }
}