package com.example.emotionapp;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;


import android.app.AlertDialog;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.camerakit.CameraKitView;
import com.example.emotionapp.Helper.GraphicOverlay;
import com.example.emotionapp.Helper.RectOverlay;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import dmax.dialog.SpotsDialog;

public class MainActivity extends AppCompatActivity {

    private Button faceDetectButton;
    private GraphicOverlay graphicOverlay;
    private CameraKitView cameraKitView;
    private AlertDialog alertDialog;
    private Interpreter interpreter;
    private Button predictButton;
    private Bitmap faceImage;
    private Rect facePosition;
    private List<String> labels;
    private TextView predictionText;

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        faceDetectButton = findViewById(R.id.detect_face_btn);
        graphicOverlay = findViewById(R.id.graphic_overlay);
        cameraKitView = findViewById(R.id.camera_view);
        predictButton = findViewById(R.id.predict_emotion_btn);
        predictionText = findViewById(R.id.prediction_text);
        alertDialog = new SpotsDialog.Builder()
                .setContext(this)
                .setMessage("Processing")
                .setCancelable(false)
                .build();

        try {
            long timeStart = System.currentTimeMillis();
            interpreter = new Interpreter(loadModelFile());
            long timeEnd = System.currentTimeMillis();
            System.out.println("Load: " + (timeEnd - timeStart));
        } catch (Exception e) {
            e.printStackTrace();
        }

        faceDetectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraKitView.startVideo();
                cameraKitView.captureImage(new CameraKitView.ImageCallback() {
                    @Override
                    public void onImage(CameraKitView cameraKitView, byte[] bytes) {
                          alertDialog.show();
                        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                        bitmap = Bitmap
                                .createScaledBitmap(bitmap, cameraKitView.getWidth(),
                                        cameraKitView.getHeight(), false);
                        cameraKitView.stopVideo();
                        faceImage = bitmap;

                        processFaceDetection(bitmap);
                    }
                });
                graphicOverlay.clear();
            }
        });

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                predictEmotion();
            }
        });
    }

    private void predictEmotion() {

        int imageTensorIndex = 0;
        int[] imageShape = interpreter.getInputTensor(imageTensorIndex).shape();
        int imageSizeY = imageShape[1];
        int imageSizeX = imageShape[2];
        DataType imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType();

        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                interpreter.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();

        TensorImage inputImageBuffer = new TensorImage(imageDataType);
        TensorBuffer outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();

        Bitmap croppedImage = Bitmap.createBitmap(faceImage, facePosition.left,
                facePosition.top, facePosition.width(), facePosition.height());
        Bitmap scaledImage = Bitmap.createScaledBitmap(croppedImage, imageSizeX, imageSizeY, true);

        if(imageShape[imageShape.length - 1] == 3) {
            inputImageBuffer.load(scaledImage);

//            ImageProcessor imageProcessor =
//                    new ImageProcessor.Builder()
//                            .add(new ResizeWithCropOrPadOp(imageSizeX, imageSizeY))
//                            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
//                            .add(new NormalizeOp(0.0f, 1.0f))
//                            .build();
//            inputImageBuffer = imageProcessor.process(inputImageBuffer);
            long timeStart = System.currentTimeMillis();
            interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
            long timeEnd = System.currentTimeMillis();
            System.out.println("Predict: " + (timeEnd - timeStart));
        } else {
            byte[] grayImage = java_convertIMG2GreyArray(scaledImage);
            ByteBuffer imgData = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY);
            imgData.order(ByteOrder.nativeOrder());
            imgData.rewind();
            imgData.put(grayImage);
            long timeStart = System.currentTimeMillis();
            interpreter.run(imgData, outputProbabilityBuffer.getBuffer().rewind());
            long timeEnd = System.currentTimeMillis();
            System.out.println("Predict: " + (timeEnd - timeStart));
        }

        try {
            labels = FileUtil.loadLabels(MainActivity.this, "labelsMobNet.txt");
        } catch (Exception e) {
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap = (Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> value : labeledProbability.entrySet()) {
            if(value.getValue() == maxValueInMap) {
                predictionText.setText(value.getKey());
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mobnet_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private byte[] java_convertIMG2GreyArray(Bitmap img) {

        byte[] theBytes = null;
        /* Get the width and height of the bitmap*/
        int width = img.getWidth();
        int height = img.getHeight();
        /* Get the pixels of the bitmap*/
        int[] pixels = new int[width * height];
        img.getPixels(pixels, 0, width, 0, 0, width, height);
        /* define the result data array*/
        theBytes = new byte[width * height/2];
        /*define Variables used in the loop, saving memory and time*/
        int x, y, k;
        int pixel, r, g, b;
        for (y = 0; y <height; y++) {
            for (x = 0, k = 0; x <width; x++) {
                //Get pixels in turn
                pixel = pixels[y * width + x];
                //Get rgb
                r = (pixel >> 16) & 0xFF;
                g = (pixel >> 8) & 0xFF;
                b = pixel & 0xFF;
                /*Save every two lines as one line*/
                if (x% 2 == 1) {
                    theBytes[k + y * width/2] = (byte ) (theBytes[k + y
                            * width/2] | ((r * 299 + g * 587 + b * 114 + 500)/1000) & 0xf0);
                    k++;
                } else {
                    theBytes[k + y * width/2 ] = (byte) (theBytes[k + y
                            * width/2] | (((r * 299 + g * 587 + b * 114 + 500)/1000) >> 4) & 0x0f);
                }
            }
        }
        return theBytes ;
    }

    private void processFaceDetection(Bitmap bitmap) {

        FirebaseVisionImage firebaseVisionImage = FirebaseVisionImage.fromBitmap(bitmap);

        FirebaseVisionFaceDetectorOptions firebaseVisionFaceDetectorOptions =
                new FirebaseVisionFaceDetectorOptions.Builder().build();

        FirebaseVisionFaceDetector firebaseVisionFaceDetector = FirebaseVision.getInstance()
                .getVisionFaceDetector(firebaseVisionFaceDetectorOptions);

        firebaseVisionFaceDetector.detectInImage(firebaseVisionImage)
                .addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionFace>>() {
                    @Override
                    public void onSuccess(List<FirebaseVisionFace> firebaseVisionFaces) {
                        getFaceResult(firebaseVisionFaces);
                    }
                }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Toast.makeText(MainActivity.this, "Error: " + e, Toast.LENGTH_SHORT).show();
            }
        });

    }

    private void getFaceResult(List<FirebaseVisionFace> firebaseVisionFaces) {

        for (FirebaseVisionFace face : firebaseVisionFaces) {
            Rect rect = face.getBoundingBox();
            RectOverlay rectOverlay = new RectOverlay(graphicOverlay, rect);
            facePosition = face.getBoundingBox();
            graphicOverlay.add(rectOverlay);
        }
        alertDialog.dismiss();
    }

    @Override
    protected void onStart() {
        super.onStart();
        cameraKitView.onStart();
    }

    @Override
    protected void onPause() {
        super.onPause();

        cameraKitView.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraKitView.onResume();
    }

    @Override
    protected void onStop() {
        super.onStop();
        cameraKitView.onStop();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        cameraKitView.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}