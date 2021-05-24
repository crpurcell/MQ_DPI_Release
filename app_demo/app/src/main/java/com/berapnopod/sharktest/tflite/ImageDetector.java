package com.berapnopod.sharktest.tflite;

import android.content.res.AssetManager;

import java.io.IOException;

public class ImageDetector {
    private static final String MODEL_FILE = "detect.tflite";
    private static final String LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final int INPUT_SIZEX = 800;
    private static final int INPUT_SIZEY = 450;
    private static final boolean IS_QUANTIZED = true;

    private ImageDetector() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     */
    public static Detector create(AssetManager assetManager, int width, int height) throws IOException {
        final Detector imageDetector = TFLiteObjectDetectionAPIModel.create(
                assetManager,
                MODEL_FILE,
                LABELS_FILE,
                INPUT_SIZEX,
                INPUT_SIZEY,
                IS_QUANTIZED);
        ((TFLiteObjectDetectionAPIModel) imageDetector).setOriginalImageSize(width, height);
        return imageDetector;
    }
}

