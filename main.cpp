#include <windows.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstring> // для strncmp

using namespace std;

const int BITRATE_TABLE[3][4][16] = {
    // MPEG1 (версия 3)
    {
        {0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0}, // Layer I
        {0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0},   // Layer II
        {0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0}     // Layer III
    },
    // MPEG2 (версия 2)
    {
        {0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0},   // Layer I
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},        // Layer II
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0}         // Layer III
    },
    // MPEG2.5 (версия 0)
    {
        {0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0},   // Layer I
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},       // Layer II
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0}        // Layer III
    }
};

const int SAMPLE_RATE_TABLE[4][4] = {
    {11025, 12000, 8000, 0},   // MPEG2.5
    {0, 0, 0, 0},              // Reserved
    {22050, 24000, 16000, 0},  // MPEG2
    {44100, 48000, 32000, 0}   // MPEG1
};

string GetMP3FilePath() {
    OPENFILENAME ofn;
    char szFile[260] = {0};
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "MP3 Files\0*.mp3\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    return GetOpenFileName(&ofn) ? szFile : "";
}

// Функция для анализа ID3v2-тега на наличие скрытых данных
void AnalyzeID3v2Tag(ifstream &file) {
    // Сохраняем исходную позицию (обычно 0)
    streampos originalPos = file.tellg();
    file.seekg(0, ios::beg);

    char header[10];
    file.read(header, 10);
    if (strncmp(header, "ID3", 3) != 0) {
        cout << "ID3v2 тег не обнаружен.\n";
        file.seekg(originalPos, ios::beg);
        return;
    }

    // Извлекаем версию, ревизию, флаги и размер тега
    int ver = static_cast<unsigned char>(header[3]);     // Основная версия (например, 3 для ID3v2.3)
    int rev = static_cast<unsigned char>(header[4]);       // Ревизия
    unsigned char flags = header[5];
    // Размер тега записан в виде синхсейф-числа (7 бит на байт)
    uint32_t tagSize = ((header[6] & 0x7F) << 21) |
                       ((header[7] & 0x7F) << 14) |
                       ((header[8] & 0x7F) << 7)  |
                       (header[9] & 0x7F);

    cout << "Обнаружен ID3v2 тег:\n";
    cout << "  Версия: 2." << ver << "." << rev << "\n";
    cout << "  Флаги: 0x" << hex << (int)flags << dec << "\n";
    cout << "  Размер тега: " << tagSize << " байт\n";

    // Читаем все данные тега
    vector<unsigned char> tagData(tagSize);
    file.read(reinterpret_cast<char*>(tagData.data()), tagSize);

    cout << "\nАнализ фреймов ID3v2-тега:\n";
    size_t pos = 0;
    // Для ID3v2.3 и ID3v2.4 заголовок фрейма имеет длину 10 байт:
    // 4 байта - идентификатор фрейма, 4 байта - размер, 2 байта - флаги.
    while (pos + 10 <= tagData.size()) {
        char frameID[5] = {0};
        memcpy(frameID, tagData.data() + pos, 4);
        // Если идентификатор равен нулю – достигнут паддинг
        if (frameID[0] == '\0')
            break;

        uint32_t frameSize = 0;
        if (ver == 3) {
            // ID3v2.3: размер – 4 байта big-endian
            frameSize = (static_cast<unsigned char>(tagData[pos+4]) << 24) |
                        (static_cast<unsigned char>(tagData[pos+5]) << 16) |
                        (static_cast<unsigned char>(tagData[pos+6]) << 8)  |
                        (static_cast<unsigned char>(tagData[pos+7]));
        } else if (ver == 4) {
            // ID3v2.4: размер – 4 байта синхсейф
            frameSize = ((tagData[pos+4] & 0x7F) << 21) |
                        ((tagData[pos+5] & 0x7F) << 14) |
                        ((tagData[pos+6] & 0x7F) << 7)  |
                        (tagData[pos+7] & 0x7F);
        } else {
            // Если версия другая – пропускаем анализ
            break;
        }

        // Флаги (2 байта, можно использовать при необходимости)
        uint16_t frameFlags = (static_cast<unsigned char>(tagData[pos+8]) << 8) |
                              static_cast<unsigned char>(tagData[pos+9]);

        cout << "  Фрейм: " << frameID 
             << " | Размер: " << frameSize << " байт"
             << " | Флаги: 0x" << hex << frameFlags << dec << "\n";

        // Если размер фрейма нулевой или выходит за пределы данных – прекращаем анализ
        if (frameSize == 0 || pos + 10 + frameSize > tagData.size())
            break;


        pos += 10 + frameSize;
    }
    if (pos < tagData.size()) {
        size_t paddingSize = tagData.size() - pos;
        cout << "  (неиспользованные байты) в конце тега: " << paddingSize << " байт\n";
        size_t totalBits = paddingSize * 8;
        size_t onesCount = 0;
        for (size_t i = pos; i < tagData.size(); i++) {
            for (int bit = 0; bit < 8; bit++) {
                if (tagData[i] & (1 << bit))
                    onesCount++;
            }
        }
        cout << "  Паддинг: " << totalBits << " бит, "
             << onesCount << " единиц, " << (totalBits - onesCount) << " нулей\n";
    }

    // Возвращаем указатель файла в исходное состояние для дальнейшего анализа MP3 фреймов
    file.clear();
    file.seekg(originalPos, ios::beg);
}

void SkipID3v2(ifstream &file) {
    char header[10];
    file.read(header, 10);
    if (!strncmp(header, "ID3", 3)) {
        uint32_t size = ((header[6] & 0x7F) << 21) |
                        ((header[7] & 0x7F) << 14) |
                        ((header[8] & 0x7F) << 7)  |
                        (header[9] & 0x7F);
        file.seekg(size + 10, ios::beg);
    } else {
        file.seekg(0, ios::beg);
    }
}

// Модифицированная функция парсинга MP3 фреймов, которая также анализирует биты заголовков.
vector<unsigned char> ParseMP3Frames(ifstream &file,
                                     int &totalFrames,
                                     int &privateBitCount,
                                     int &copyrightCount,
                                     int &originalCount) {
    vector<unsigned char> hidden;
    streampos lastPos = file.tellg();

    totalFrames = 0;
    privateBitCount = 0;
    copyrightCount = 0;
    originalCount = 0;

    while (file) {
        streampos start = file.tellg();
        unsigned char header[4];
        file.read(reinterpret_cast<char*>(header), 4);

        if (header[0] != 0xFF || (header[1] & 0xE0) != 0xE0) {
            file.seekg(start + streampos(1));
            continue;
        }

        int mpegVersion = (header[1] >> 3) & 0x03;
        int layer = (header[1] >> 1) & 0x03;
        int bitrateIdx = (header[2] >> 4) & 0x0F;
        int sampleRateIdx = (header[2] >> 2) & 0x03;
        int padding = (header[2] >> 1) & 0x01;

        if (mpegVersion == 1 || layer == 0 || bitrateIdx == 0 || bitrateIdx == 15) {
            file.seekg(start + streampos(1));
            continue;
        }

        int versionIdx = (mpegVersion == 3) ? 0 : (mpegVersion == 2) ? 1 : 2;
        int layerIdx = (layer == 3) ? 0 : (layer == 2) ? 1 : 2;
        int bitrate = BITRATE_TABLE[versionIdx][layerIdx][bitrateIdx] * 1000;
        int sampleRate = SAMPLE_RATE_TABLE[mpegVersion][sampleRateIdx];
        if (!bitrate || !sampleRate) {
            file.seekg(start + streampos(1));
            continue;
        }

        // Анализ битов заголовка MP3
        int privateBit = header[2] & 0x01;
        int copyrightBit = (header[3] >> 3) & 0x01;
        int originalBit = (header[3] >> 2) & 0x01;

        totalFrames++;
        if (privateBit) privateBitCount++;
        if (copyrightBit) copyrightCount++;
        if (originalBit) originalCount++;

        int frameLength = (layerIdx == 0) ?
            (12 * bitrate / sampleRate + padding) * 4 :
            (144 * bitrate / sampleRate) + padding;
        if (frameLength <= 0) {
            file.seekg(start + streampos(1));
            continue;
        }
        // Собираем данные между предыдущим фреймом и текущим (скрытые данные)
        streampos gapSize = start - lastPos;
        if (gapSize > 0) {
            file.seekg(lastPos);
            vector<unsigned char> gap(gapSize);
            file.read(reinterpret_cast<char*>(gap.data()), gapSize);
            hidden.insert(hidden.end(), gap.begin(), gap.end());
        }

        lastPos = start + streampos(frameLength);
        file.seekg(lastPos);
    }
    file.clear();
    file.seekg(0, ios::end);
    streampos endPos = file.tellg();
    if (endPos > lastPos) {
        file.seekg(lastPos);
        vector<unsigned char> endData(endPos - lastPos);
        file.read(reinterpret_cast<char*>(endData.data()), endData.size());

        if (endData.size() >= 128 && string(endData.end() - 128, endData.end() - 125) == "TAG") {
            endData.resize(endData.size() - 128);
        }
        hidden.insert(hidden.end(), endData.begin(), endData.end());
    }
    return hidden;
}
void AnalyzeHiddenData(const vector<unsigned char>& hidden) {
    if (hidden.empty()){
        cout << "Не найдено вспомогательных данных (Ancillary Data).\n";
        return;
    }
    size_t totalBits = hidden.size() * 8;
    size_t onesCount = 0;
    for (auto byte : hidden) {
        for (int i = 0; i < 8; i++) {
            if (byte & (1 << i))
                onesCount++;
        }
    }
    cout << "\nАнализ вспомогательных данных (Ancillary Data):\n";
    cout << "Общее количество бит: " << totalBits << "\n";
    cout << "Количество единиц: " << onesCount << " (" << fixed << setprecision(2)
         << (100.0 * onesCount / totalBits) << "%)\n";
    cout << "Количество нулей: " << (totalBits - onesCount) << " (" << fixed << setprecision(2)
         << (100.0 * (totalBits - onesCount) / totalBits) << "%)\n";
}

void ScanForSignatures(const vector<unsigned char>& hiddenData) {
    vector<vector<unsigned char>> signatures = {
        {0x00, 0xFF, 0xD9}, // QuickCrypto
        {0x00, 0x00, 0x00, 0x1E}, // MP3Sponge
        {0x0A, 0x0A}, // File Mask
        {0x26, 0x29}, // File Mask
        {0x28, 0x24, 0x23, 0x5E, 0x40, 0x2A, 0x23, 0x5E, 0x28, 0x00} // DeEgger
    };

    vector<string> signatureNames = {
        "QuickCrypto (00 FF D9)",
        "MP3Sponge (00 00 00 1E)",
        "File Mask (0A 0A)",
        "File Mask (26 29)",
        "DeEgger (28 24 23 5E 40 2A 23 5E 28 00)"
    };

    for (size_t i = 0; i < signatures.size(); ++i) {
        const vector<unsigned char>& signature = signatures[i];
        const string& signatureName = signatureNames[i];

        bool found = false;
        for (size_t j = 0; j + signature.size() <= hiddenData.size(); ++j) {
            if (equal(signature.begin(), signature.end(), hiddenData.begin() + j)) {
                found = true;
                cout << "Найдена сигнатура " << signatureName << " по смещению " << hex << j << endl;
            }
        }
        if (!found) {
            cout << "Сигнатура " << signatureName << " не найдена." << endl;
        }
    }
}

void AnalyzeAdditionalBit(const vector<unsigned char>& hidden) {
    if (hidden.empty()){
        cout << "Нет данных для анализа добавочного бита.\n";
        return;
    }
    int additionalBit = hidden[0] & 0x01;
    cout << "\nАнализ добавочного бита:\n";
    cout << "Добавочный бит (LSB первого байта скрытых данных): " << additionalBit << "\n";
}

int main() {
    string path = GetMP3FilePath();
    if (path.empty()) {
        cout << "Файл не выбран.\n";
        return 0;
    }

    ifstream file(path, ios::binary);
    if (!file) {
        cout << "Ошибка открытия файла.\n";
        return 0;
    }
    // Анализ ID3v2-тега на наличие скрытых данных
    AnalyzeID3v2Tag(file);
    // Пропускаем ID3v2 тег для дальнейшего анализа MP3 фреймов
    SkipID3v2(file);
    int totalFrames = 0, privateBitCount = 0, copyrightCount = 0, originalCount = 0;
    vector<unsigned char> hiddenData = ParseMP3Frames(file,
                                                      totalFrames,
                                                      privateBitCount,
                                                      copyrightCount,
                                                      originalCount);
    cout << "\nАнализ заголовков MP3 фреймов:\n";
    cout << "Всего фреймов: " << totalFrames << "\n";
    cout << "Битов Private установлено: " << privateBitCount << "\n";
    cout << "Битов Copyright установлено: " << copyrightCount << "\n";
    cout << "Битов Original/Home установлено: " << originalCount << "\n";
    cout << "\nСкрытые данные (" << hiddenData.size() << " байт):\n";
    for (size_t i = 0; i < hiddenData.size(); ++i) {
        cout << hex << setw(2) << setfill('0') << (int)hiddenData[i] << " ";
        if ((i + 1) % 16 == 0)
            cout << endl;
    }
    cout << dec << endl;
    AnalyzeHiddenData(hiddenData);
    AnalyzeAdditionalBit(hiddenData);
    ScanForSignatures(hiddenData);
    return 0;
}
