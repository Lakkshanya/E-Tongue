#include <OneWire.h>
#include <DallasTemperature.h>

#define PH_PIN A0
#define TDS_PIN A1
#define ONE_WIRE_BUS 2

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

bool sentReading = false; // send only once per solution

void setup() {
  Serial.begin(9600);
  sensors.begin();
}

void loop() {
  // Read temperature
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  // Read raw analog values
  int phRaw = analogRead(PH_PIN);
  int tdsRaw = analogRead(TDS_PIN);

  // Convert to voltage (10-bit ADC)
  float phVoltage = phRaw * (5.0 / 1023.0);
  float tdsVoltage = tdsRaw * (5.0 / 1023.0);

  // --- pH calibration ---
  float ph4Voltage = 3.0;
  float ph7Voltage = 2.5;
  float slope = (7.0 - 4.0) / (ph7Voltage - ph4Voltage);
  float intercept = 7.0 - slope * ph7Voltage;
  float pH = slope * phVoltage + intercept;

  // --- TDS calculation with temperature compensation ---
  float compensationCoefficient = 1.0 + 0.02 * (tempC - 25.0);
  float compensatedVoltage = tdsVoltage / compensationCoefficient;
  float tds = (133.42 * pow(compensatedVoltage, 3)
             - 255.86 * pow(compensatedVoltage, 2)
             + 857.39 * compensatedVoltage) * 0.5;

  // Only send once per solution
  if (!sentReading) {
    // Prepare JSON
    String jsonData = "{";
    jsonData += "\"Hardness\":180,";
    jsonData += "\"Solids_TDS\":" + String(tds) + ",";
    jsonData += "\"Sulphate\":320,";
    jsonData += "\"Chloramine\":6.5,";
    jsonData += "\"Conductivity\":400,";
    jsonData += "\"Organic_Carbon\":7,";
    jsonData += "\"Trihalomethane\":70,";
    jsonData += "\"Turbidity\":4,";
    jsonData += "\"pH\":" + String(pH) + ",";
    jsonData += "\"temperature\":" + String(tempC);
    jsonData += "}";

    Serial.println(jsonData);
    sentReading = true; // prevent sending again
  }

  delay(2000); // wait before next loop (optional)
}
