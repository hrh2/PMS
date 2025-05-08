#include <SPI.h>
#include <MFRC522.h>

#define RST_PIN  9 
#define SS_PIN   10  

MFRC522 mfrc522(SS_PIN, RST_PIN);
MFRC522::MIFARE_Key key;
MFRC522::StatusCode card_status;

void setup(){
  Serial.begin(9600);        
  SPI.begin();
  mfrc522.PCD_Init();
  Serial.println(F("==== CARD REGISTRATION ===="));
  Serial.println(F("Place your RFID card near the reader..."));
  Serial.println();
}

void loop(){
  // Set default key
  for(byte i = 0; i < 6; i++){ 
    key.keyByte[i] = 0xFF;
  }

  // Wait for card
  if(!mfrc522.PICC_IsNewCardPresent()) return;
  if (!mfrc522.PICC_ReadCardSerial()) return;

  Serial.println(F("Card detected!"));

  // Buffer for writing
  byte carPlateBuff[16];
  byte balanceBuff[16];

  // Request car plate
  Serial.println(F("Enter car plate number (7 characters, end with # press ENTER):"));
  Serial.setTimeout(20000L); // Wait up to 20 seconds
  byte len = Serial.readBytesUntil('#', (char *)carPlateBuff, 16);

  if (len != 7) {
    Serial.println(F("❌ Invalid car plate. Must be exactly 7 characters (e.g., RAG234H). Try again.\n"));
    mfrc522.PICC_HaltA();
    mfrc522.PCD_StopCrypto1();
    delay(2000);
    return;
  }

  padBuffer(carPlateBuff, len);


  // Request balance
  Serial.println(F("Enter balance (end with # press ENTER):"));
  len = Serial.readBytesUntil('#', (char *)balanceBuff, 16);
  padBuffer(balanceBuff, len);

  // Write to RFID blocks
  byte carPlateBlock = 2;
  byte balanceBlock = 4;

  writeBytesToBlock(carPlateBlock, carPlateBuff);
  writeBytesToBlock(balanceBlock, balanceBuff);

  Serial.println();
  Serial.print(F("✅ Car Plate written to block "));
  Serial.print(carPlateBlock);
  Serial.print(F(": "));
  Serial.println((char*)carPlateBuff);

  Serial.print(F("✅ Balance written to block "));
  Serial.print(balanceBlock);
  Serial.print(F(": "));
  Serial.println((char*)balanceBuff);

  Serial.println(F("Please remove the card to write again."));
  Serial.println(F("--------------------------"));
  Serial.println();

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  delay(2000);
}

void padBuffer(byte* buffer, byte len) {
  for(byte i = len; i < 16; i++){
    buffer[i] = ' '; // Pad with spaces
  }
}

void writeBytesToBlock(byte block, byte buff[]){
  card_status = mfrc522.PCD_Authenticate(MFRC522::PICC_CMD_MF_AUTH_KEY_A, block, &key, &(mfrc522.uid));
  
  if(card_status != MFRC522::STATUS_OK) {
    Serial.print(F("❌ Authentication failed: "));
    Serial.println(mfrc522.GetStatusCodeName(card_status));
    return;
  } else {
    Serial.println(F("🔓 Authentication success."));
  }

  card_status = mfrc522.MIFARE_Write(block, buff, 16);
  if (card_status != MFRC522::STATUS_OK) {
    Serial.print(F("❌ Write failed: "));
    Serial.println(mfrc522.GetStatusCodeName(card_status));
    return;
  } else {
    Serial.println(F("✅ Data written successfully."));
  }
}
