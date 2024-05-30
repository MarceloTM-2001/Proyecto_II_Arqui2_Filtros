#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// Estructura para el encabezado de un archivo BMP
#pragma pack(push, 1)
typedef struct {
    unsigned short type;       // Tipo de archivo (debe ser 'BM' para un archivo BMP válido)
    unsigned int size;         // Tamaño del archivo en bytes
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int offset;       // Desplazamiento a los datos de píxeles
} BMPHeader;

// Estructura para el encabezado de información de imagen BMP
typedef struct {
    unsigned int size;           // Tamaño de esta estructura en bytes
    int width;                   // Ancho de la imagen en píxeles
    int height;                  // Altura de la imagen en píxeles
    unsigned short planes;       // Número de planos de color (debe ser 1)
    unsigned short bitCount;     // Número de bits por píxel
    unsigned int compression;    // Método de compresión utilizado
    unsigned int imageSize;      // Tamaño de los datos de imagen en bytes
    int xPixelsPerMeter;         // Resolución horizontal en píxeles por metro
    int yPixelsPerMeter;         // Resolución vertical en píxeles por metro
    unsigned int colorsUsed;     // Número de colores en la paleta
    unsigned int colorsImportant;// Número de colores importantes
} BMPInfoHeader;
#pragma pack(pop)

void printHeaders(BMPHeader *header, BMPInfoHeader *infoHeader) {
    printf("BMP Header:\n");
    printf("  Type: %u\n", header->type);
    printf("  Size: %u\n", header->size);
    printf("  Reserved1: %u\n", header->reserved1);
    printf("  Reserved2: %u\n", header->reserved2);
    printf("  Offset: %u\n", header->offset);

    printf("BMP Info Header:\n");
    printf("  Size: %u\n", infoHeader->size);
    printf("  Width: %d\n", infoHeader->width);
    printf("  Height: %d\n", infoHeader->height);
    printf("  Planes: %u\n", infoHeader->planes);
    printf("  BitCount: %u\n", infoHeader->bitCount);
    printf("  Compression: %u\n", infoHeader->compression);
    printf("  ImageSize: %u\n", infoHeader->imageSize);
    printf("  XPixelsPerMeter: %d\n", infoHeader->xPixelsPerMeter);
    printf("  YPixelsPerMeter: %d\n", infoHeader->yPixelsPerMeter);
    printf("  ColorsUsed: %u\n", infoHeader->colorsUsed);
    printf("  ColorsImportant: %u\n", infoHeader->colorsImportant);
}

void gray_conversion(unsigned char *data, BMPInfoHeader infoHeader, unsigned char *newdata, int start, int end) {
    int width = infoHeader.width;
    int rowSize = width * 3;

    for (int y = start; y < end; y++) {
        for (int x = 0; x < width; x++) {
            int pos = y * rowSize + x * 3;
            unsigned char newcolor = (unsigned char)(0.3 * data[pos + 2] + 0.59 * data[pos + 1] + 0.11 * data[pos]);
            newdata[pos] = newcolor;
            newdata[pos + 1] = newcolor;
            newdata[pos + 2] = newcolor;
        }
    }
}

void blur_conversion(unsigned char *data, BMPInfoHeader infoHeader, unsigned char *newdata, int start, int end) {
    float kernel[3][3] = {
            {1.0/16, 2.0/16, 1.0/16},
            {2.0/16, 4.0/16, 2.0/16},
            {1.0/16, 2.0/16, 1.0/16}
    };

    int width = infoHeader.width;
    for (int y = start; y < end; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0.0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int pos = ((y + i) * width + (x + j)) * 3 + c;
                        sum += kernel[i + 1][j + 1] * data[pos];
                    }
                }
                newdata[(y * width + x) * 3 + c] = (unsigned char)sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BMPHeader bmpHeader;
    BMPInfoHeader bmpInfoHeader;
    unsigned char *data = NULL;

    if (rank == 0) {
        FILE *inputFile = fopen("View.bmp", "rb");
        if (inputFile == NULL) {
            perror("Error abriendo el archivo de entrada");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fread(&bmpHeader, sizeof(BMPHeader), 1, inputFile);
        if (bmpHeader.type != 0x4D42) { // Verificar que el archivo es un BMP ('BM')
            printf("El archivo no es un BMP válido\n");
            fclose(inputFile);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fread(&bmpInfoHeader, sizeof(BMPInfoHeader), 1, inputFile);
        printHeaders(&bmpHeader, &bmpInfoHeader);

        data = (unsigned char *)malloc(bmpInfoHeader.imageSize);
        if (data == NULL) {
            fprintf(stderr, "No se pudo asignar memoria para los datos de la imagen.\n");
            fclose(inputFile);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fread(data, 1, bmpInfoHeader.imageSize, inputFile);
        fclose(inputFile);
    }

    // Broadcast the BMP header and info header to all nodes
    MPI_Bcast(&bmpHeader, sizeof(BMPHeader), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bmpInfoHeader, sizeof(BMPInfoHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

    int height = bmpInfoHeader.height;
    int width = bmpInfoHeader.width;
    int rowSize = width * 3;
    int localHeight = height / size;
    int localSize = localHeight * rowSize;

    unsigned char *subData = (unsigned char *)malloc(localSize);
    unsigned char *subDataProcessed = (unsigned char *)malloc(localSize);

    MPI_Scatter(data, localSize, MPI_UNSIGNED_CHAR, subData, localSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Calcular el número de filas que cada proceso debe procesarstart
    int rows_per_process = height / size;
    int remaining_rows = height % size;

    // Ajustar los límites para el último proceso
    if (rank == size - 1) {
        // El último proceso toma las filas restantes
        localHeight = rows_per_process + remaining_rows;
    } else {
        // Los demás procesos toman el número igual de filas
        localHeight = rows_per_process;
    }

    // Calcular los límites para el procesamiento
    int start = rank * rows_per_process;
    int end = start + localHeight;
    if(!strcmp(argv[1],"grey")){
        if(rank==0){
            printf("Hola soy master trabajando\n");
            gray_conversion(subData, bmpInfoHeader, subDataProcessed, 0, localHeight); 
        }else if (rank == 1) {
            printf("Hola soy esclavo grey 1\n");
            gray_conversion(subData, bmpInfoHeader, subDataProcessed, 0, localHeight);
        } else if (rank == 2) {
            printf("Hola soy esclavo grey 2\n");
            gray_conversion(subData, bmpInfoHeader, subDataProcessed, 0, localHeight);
        } else if (rank==3){
            printf("Hola soy esclavo grey 3\n");
            gray_conversion(subData, bmpInfoHeader, subDataProcessed, 0, localHeight);
        }else if (rank==4){
            printf("Hola soy esclavo grey 4\n");
            gray_conversion(subData, bmpInfoHeader, subDataProcessed, 0, localHeight);
        }else if (rank==5){
            printf("Hola soy esclavo grey 5\n");
            gray_conversion(subData, bmpInfoHeader, subDataProcessed, 0, localHeight);
        }else {
            memcpy(subDataProcessed, subData, localSize);
        }
    }else{
        if(rank==0){
            printf("Hola soy master trabajando\n");
            blur_conversion(subData, bmpInfoHeader, subDataProcessed, 1, localHeight-1);
        }else if(rank==1){
            printf("Hola soy esclavo blur 1\n");
            blur_conversion(subData, bmpInfoHeader, subDataProcessed, 1, localHeight-1);
        } else if (rank == 2) {
            printf("Hola soy esclavo blur 2\n");
            blur_conversion(subData, bmpInfoHeader, subDataProcessed, 1, localHeight-1);
        } else if (rank==3){
            printf("Hola soy esclavo blur 3\n");
            blur_conversion(subData, bmpInfoHeader, subDataProcessed, 1, localHeight-1);
        }else if (rank==4){
            printf("Hola soy esclavo blur 4\n");
            blur_conversion(subData, bmpInfoHeader, subDataProcessed, 1, localHeight-1);
        }else if (rank==5){
            printf("Hola soy esclavo blur 5\n");
            blur_conversion(subData, bmpInfoHeader, subDataProcessed, 1, localHeight-1);
        }else{
            memcpy(subDataProcessed, subData, localSize);
        }
    }
    unsigned char *newData = NULL;
    if (rank == 0) {
        newData = (unsigned char *)malloc(bmpInfoHeader.imageSize);
    }
    MPI_Gather(subDataProcessed, localSize, MPI_UNSIGNED_CHAR, newData, localSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *outputFile = fopen("Processed_Image.bmp", "wb");
        if (!outputFile) {
            printf("No se pudo crear el archivo de salida\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fwrite(&bmpHeader, sizeof(BMPHeader), 1, outputFile);
        fwrite(&bmpInfoHeader, sizeof(BMPInfoHeader), 1, outputFile);
        fwrite(newData, 1, bmpInfoHeader.imageSize, outputFile);

        fclose(outputFile);
        free(newData);
    }
    if (rank == 0) {
        free(subData);
        free(subDataProcessed);
        free(data);
    }

    MPI_Finalize();
    return 0;
}

