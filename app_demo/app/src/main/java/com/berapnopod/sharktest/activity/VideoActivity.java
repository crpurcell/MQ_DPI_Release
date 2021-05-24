package com.berapnopod.sharktest.activity;

import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.TextureView;
import android.view.View;
import android.widget.Toast;

import com.berapnopod.sharktest.tflite.DetectorThread;


import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

import com.berapnopod.sharktest.customview.OverlayView;
import com.berapnopod.sharktest.SharkApp;
import com.berapnopod.sharktest.event.ConnectivityChangeEvent;
import com.berapnopod.sharktest.event.ToastEvent;
import com.berapnopod.sharkdemo.R;
import dji.common.camera.SettingsDefinitions;
import dji.common.error.DJIError;
import dji.common.product.Model;
import dji.common.util.CommonCallbacks;
import dji.sdk.camera.Camera;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.products.Aircraft;
import dji.ux.panel.CameraSettingExposurePanel;
import dji.ux.widget.controls.ExposureSettingsMenu;

public class VideoActivity extends EventBusActivity implements TextureView.SurfaceTextureListener {
    private DJICodecManager djiCodecManager;
    private VideoFeeder.VideoDataCallback videoDataCallback = null;
    private TextureView videoTextureView = null;
    private DetectorThread detectorThread = null;

    private ExposureSettingsMenu exposureButton;
    private CameraSettingExposurePanel exposurePanel;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video);

        initializeViews();
        initializeExposureButton();
    }

    @Override
    public void onResume() {
        super.onResume();
        onConnectionChanged();
    }

    @Override
    public void onSurfaceTextureAvailable(final SurfaceTexture surface, int width, int height) {
        if (djiCodecManager == null) {
            djiCodecManager = new DJICodecManager(getApplicationContext(), surface, width, height);
        }

        if(videoTextureView != null) {
            OverlayView trackingOverlay = findViewById(R.id.tracking_overlay_view);

            try {
                detectorThread = new DetectorThread(videoTextureView, trackingOverlay, getAssets());
            } catch (Exception e) {
                e.printStackTrace();
            }

            if(detectorThread != null && videoTextureView != null) {
                detectorThread.start();
            }
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        if (djiCodecManager != null) {
            djiCodecManager.cleanSurface();
            djiCodecManager = null;
        }
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onConnectivityChangeEvent(ConnectivityChangeEvent event) {
        onConnectionChanged();
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onToastEventListener(ToastEvent event) {
        Toast.makeText(this, event.message, Toast.LENGTH_SHORT).show();
    }

    private void onConnectionChanged() {
        Aircraft aircraft = ((SharkApp) getApplication()).getSdkManager().getAircraftInstance();

        if (aircraft == null) {
            Toast.makeText(this, "Disconnected", Toast.LENGTH_LONG).show();
            return;
        }

        Camera camera = aircraft.getCamera();
        if (camera != null) {
            destroyPreviewer();
            camera.setMode(SettingsDefinitions.CameraMode.SHOOT_PHOTO, new CommonCallbacks.CompletionCallback() {
                @Override
                public void onResult(DJIError djiError) {
                    if (djiError != null) {
                        Toast.makeText(getApplicationContext(),
                                "Can't change mode of drone camera!",
                                Toast.LENGTH_LONG).show();
                        finish();
                    }
                }
            });
        }

        if (!aircraft.isConnected()) {
            Toast.makeText(this, "Disconnected", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        initPreviewer(aircraft);
    }

    private void initPreviewer(Aircraft aircraft) {
        if (!aircraft.getModel().equals(Model.UNKNOWN_AIRCRAFT)) {
            VideoFeeder.getInstance().getPrimaryVideoFeed().setCallback(videoDataCallback);
        }
    }

    private void destroyPreviewer() {
        VideoFeeder.getInstance().getPrimaryVideoFeed().setCallback(null);
    }

    @Override
    protected void onDestroy() {
        if(djiCodecManager != null) djiCodecManager.destroyCodec();
        super.onDestroy();
    }

    private void initializeViews() {
        videoTextureView = findViewById(R.id.video_texture_view);
        videoTextureView.setSurfaceTextureListener(this);
        exposureButton = findViewById(R.id.ExposureMenuButton);
        exposurePanel = findViewById(R.id.ExposureMenu);
    }

    private void initializeExposureButton() {
        exposureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(exposurePanel.getVisibility() != View.VISIBLE)
                    exposurePanel.setVisibility(View.VISIBLE);
                else
                    exposurePanel.setVisibility(View.INVISIBLE);
            }
        });
    }
}

