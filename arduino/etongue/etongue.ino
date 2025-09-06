#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

Adafruit_ADS1115 ads;
#define ONE_WIRE_BUS 2
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(9600);  // Serial communication
  ads.begin();
  sensors.begin();
}

void loop() {
  // Read sensors
  int16_t phRaw = ads.readADC_SingleEnded(0);
  int16_t tdsRaw = ads.readADC_SingleEnded(1);

  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  float phVoltage = phRaw * 0.1875 / 1000.0;
  float tdsVoltage = tdsRaw * 0.1875 / 1000.0;

  float pH = 7.0 + ((2.5 - phVoltage) / 0.18);
  float tds = tdsVoltage * 1000.0;

  // Prepare JSON string
  String jsonData = "{";
  jsonData += "\"Hardness\":180,";           // dummy
  jsonData += "\"Solids_TDS\":" + String(tds) + ",";
  jsonData += "\"Sulphate\":320,";           // dummy
  jsonData += "\"Chloramine\":6.5,";         // dummy
  jsonData += "\"Conductivity\":400,";       // dummy
  jsonData += "\"Organic_Carbon\":7,";       // dummy
  jsonData += "\"Trihalomethane\":70,";      // dummy
  jsonData += "\"Turbidity\":4,";            // dummy
  jsonData += "\"pH\":" + String(pH) + ",";
  jsonData += "\"temperature\":" + String(tempC);
  jsonData += "}";

  // Send JSON over Serial
  Serial.println(jsonData);

  delay(2000);  // send every 2 seconds
}
