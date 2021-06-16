#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Event.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/BoxFilter.h>


#include <vpi/OpenCVInterop.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            printf("Error %d: %s\n %s\n", __LINE__,           \
            vpiStatusGetName(status), buffer);                \
            exit(1);                                          \
        }                                                     \
    } while (0);

int main(int argc, char *argv[]){

  cv::VideoCapture camera;

  camera.open(0);

  // check if we succeeded
  if (!camera.isOpened()) {
      printf("ERROR! Unable to open camera\n");
      return -1;
  }

  cv::Mat inImage;
  camera.read(inImage);
  //inImage = cv::imread("aloeL.jpg",cv::IMREAD_COLOR);
  //cv::imshow("image0", inImage);

  //Wrap cv::Mat
  VPIImage wrapImage;
  CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(inImage, VPI_BACKEND_CPU, &wrapImage));

  //Get format
  VPIImageFormat imgFormat;
  CHECK_STATUS(vpiImageGetFormat(wrapImage, &imgFormat));

  //Allocate downscale
  VPIImage imgSmall;
  CHECK_STATUS(vpiImageCreate(480, 270, imgFormat, VPI_BACKEND_CPU,
                  &imgSmall));

  VPIImage imgGray;
  CHECK_STATUS(vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CPU,
                  &imgGray));

  VPIImage imgFilter;
  CHECK_STATUS(vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CPU,
                  &imgFilter));

  VPIStream stream;
  CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CPU, &stream));

  while(true){
    camera.read(inImage);

    vpiImageSetWrappedOpenCVMat(wrapImage, inImage);

    CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CPU, wrapImage, imgSmall,
                      VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));

    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CPU, imgSmall,
                            imgGray, NULL));

    CHECK_STATUS(vpiSubmitBoxFilter(stream, VPI_BACKEND_CPU, imgGray, imgFilter,
                          3, 3, VPI_BORDER_ZERO));

    vpiStreamSync(stream);

    VPIImageData imgdata;
    CHECK_STATUS(vpiImageLock(imgFilter, VPI_LOCK_READ, &imgdata));

    cv::Mat outFrame;
    CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata, &outFrame));

    CHECK_STATUS(vpiImageUnlock(imgFilter));

    cv::imshow("image1", outFrame);
    cv::waitKey(1);
  }


  vpiStreamDestroy(stream);

  vpiImageDestroy(wrapImage);
  vpiImageDestroy(imgSmall);

  return 0;
}
