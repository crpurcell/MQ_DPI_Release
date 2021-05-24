package com.berapnopod.sharktest.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.view.TextureView;

import com.berapnopod.sharktest.customview.OverlayView;
import com.opencsv.CSVWriter;
import com.berapnopod.sharktest.tracking.MultiBoxTracker;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

public class DetectorThread extends Thread {
    private TextureView videoTextureView;
    private OverlayView overlayView;
    private Detector detector;
    private MultiBoxTracker boxTracker;
    private int timestamp = 0;
    private FileWriter mFileWriter;

    public DetectorThread(TextureView textureView,
                          OverlayView overlayView,
                          AssetManager assetManager) throws IOException {
        this.videoTextureView = textureView;
        this.overlayView = overlayView;
        this.boxTracker = new MultiBoxTracker(videoTextureView.getContext());
        this.detector = ImageDetector.create(assetManager, videoTextureView.getWidth(), videoTextureView.getHeight());
        this.overlayView.addCallback(new OverlayView.DrawCallback() {
            @Override
            public void drawCallback(Canvas canvas) {
                boxTracker.draw(canvas);
            }
        });
    }


    public void run() {
        String baseDir = android.os.Environment.getExternalStorageDirectory().getAbsolutePath();
        String fileName = new SimpleDateFormat("yyyyMMddHHmm'.csv'").format(new Date());
        String filePath = baseDir + File.separator + fileName;
        File f = new File(filePath);
        CSVWriter writer = null;


        while (videoTextureView != null) {
            timestamp++;
            // Grab a frame, classify it, pass it to the tracker, redraw the screen
            final Bitmap currentFrame = videoTextureView.getBitmap();
            if (currentFrame == null) return;
            while (((TFLiteObjectDetectionAPIModel) detector).processing) ;
            List<Detector.Result> imageResults =
                    detector.recognizeImage(currentFrame);

            // Get a byte array from the current frame and send it to the tracker
            ByteArrayOutputStream imageOut = new ByteArrayOutputStream();
            currentFrame.compress(Bitmap.CompressFormat.JPEG, 100, imageOut);
            byte[] currentFrameBytes = imageOut.toByteArray();

            boxTracker.trackResults(imageResults, currentFrameBytes, timestamp);
            overlayView.postInvalidate();

            // File exist
            if (f.exists() && !f.isDirectory()) {
                try {
                    mFileWriter = new FileWriter(filePath, true);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                writer = new CSVWriter(mFileWriter);
            } else {
                try {
                    writer = new CSVWriter(new FileWriter(filePath));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }


            if (!imageResults.isEmpty()) {
                String timetime = new SimpleDateFormat("yyyy MMdd HH:mm:ss").format(new Date());
                String[] data = {timetime, String.valueOf(imageResults.subList(0, 1))};

                writer.writeNext(data);
            }
            try {
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
