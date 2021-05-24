package com.berapnopod.sharktest.activity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.berapnopod.sharkdemo.R;
import com.berapnopod.sharktest.SharkApp;
import com.berapnopod.sharktest.event.ConnectivityChangeEvent;
import com.berapnopod.sharktest.event.RegistrationStatusEvent;

import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

import java.util.ArrayList;
import java.util.List;

import dji.sdk.base.BaseProduct;
import dji.sdk.products.Aircraft;

/**
 * This Activity presents the main screen (entry point) of the app:
 *
 * - device (CrystalSky tablet) permission check
 * - DJI SDK registration (only required once per app installation)
 * - internet connection check (needed for initial DJI SDK registration)
 * - drone connection check
 *
 */
public class MainActivity extends EventBusActivity implements View.OnClickListener {
    private static final int REQUEST_PERMISSION_CODE = 12354;
    private static final String TAG = MainActivity.class.getSimpleName();

    private static final String[] REQUIRED_PERMISSION_LIST = new String[]{
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE,
    };

    private final List<String> missingPermissionsList = new ArrayList<>();

    private Button openCameraButton;
    private TextView registrationStatusTextView;
    private TextView droneStatusTextView;
    private TextView modelTextView;
    private RelativeLayout openProgressBarView;
    private Boolean isDroneConnected = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        registrationStatusTextView = findViewById(R.id.text_registration_status);
        openProgressBarView = findViewById(R.id.progress_open);
        droneStatusTextView = findViewById(R.id.text_connection_status);
        modelTextView = findViewById(R.id.text_model_available);

        openCameraButton = findViewById(R.id.btn_open_camera);
        openCameraButton.setOnClickListener(this);

        // Check and request permissions
        checkAndRequestPermissions();
    }

    @Override
    protected void onResume() {
        super.onResume();
        openProgressBarView.setVisibility(View.INVISIBLE);
        // Register DJI SDK
        ((SharkApp) getApplication()).getSdkManager().registerSDK();
        // Check drone connection
        checkDroneConnection();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // Check for granted permission and remove from missing list
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (int i = grantResults.length - 1; i >= 0; i--) {
                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                    missingPermissionsList.remove(permissions[i]);
                }
            }
        }
        // If there is enough permission, we will start the registration
        if (missingPermissionsList.isEmpty()) {
            // register DJI SDK
            ((SharkApp) getApplication()).getSdkManager().registerSDK();
        } else {
            Toast.makeText(this, "Missing permissions!", Toast.LENGTH_LONG).show();
        }
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onConnectivityChangeEvent(ConnectivityChangeEvent event) {
        checkDroneConnection();
    }

    // Subscribe to DJI SDK registration status
    @Subscribe(threadMode =  ThreadMode.MAIN)
    public void onRegistrationChangeEvent(RegistrationStatusEvent event) {
        // update UI with current registration status
        if(event.isSDKRegistered) {
            registrationStatusTextView.setText(R.string.dji_registration_status_registered);
        } else {
            registrationStatusTextView.setText(R.string.dji_registration_status_error);
        }
        showStatusIcon(event.isSDKRegistered, registrationStatusTextView);
    }

    /**
     * Add status icon as a Drawable to the left
     *
     * @param isChecked Is the current status OK?
     * @param textView  UI TextView
     */
    private void showStatusIcon(boolean isChecked, TextView textView) {
        if (isChecked) {
            textView.setCompoundDrawablesWithIntrinsicBounds(R.drawable.ic_checked_checkbox, 0, 0, 0);
        } else {
            textView.setCompoundDrawablesWithIntrinsicBounds(R.drawable.ic_unchecked_checkbox, 0, 0, 0);
        }
        textView.setCompoundDrawablePadding(12);
    }

    private void checkDroneConnection() {
        BaseProduct product = ((SharkApp) getApplication()).getSdkManager().getProductInstance();
        if (null != product && product.isConnected()) {
            Toast.makeText(this, "Drone Connected", Toast.LENGTH_SHORT).show();
            droneConnected(product);
        } else {
            Toast.makeText(this, "Drone Disconnected", Toast.LENGTH_LONG).show();
            droneDisconnected();
        }
    }

    private void droneConnected(BaseProduct product) {
        if (product instanceof Aircraft) {
            droneStatusTextView.setText(R.string.status_aircraft_connected);
        } else {
            droneStatusTextView.setText(R.string.status_handheld_connected);
            return;
        }
        showStatusIcon(true, droneStatusTextView); // Show drone connected status icon
        modelTextView.setText(product.getModel().getDisplayName());
        isDroneConnected = true;
        openCameraButton.setBackgroundColor(getResources().getColor(R.color.colorPrimary));
    }

    private void droneDisconnected() {
        droneStatusTextView.setText(R.string.status_drone_disconnected);
        showStatusIcon(false, droneStatusTextView); // Show drone disconnected status icon
        modelTextView.setText(R.string.drone_info_not_available);
        isDroneConnected = false;
        openCameraButton.setBackgroundColor(getResources().getColor(R.color.gray));
    }

    private void checkAndRequestPermissions() {
        for (String eachPermission : REQUIRED_PERMISSION_LIST) {
            if (ContextCompat.checkSelfPermission(this, eachPermission) != PackageManager.PERMISSION_GRANTED) {
                missingPermissionsList.add(eachPermission);
            }
        }

        if (!missingPermissionsList.isEmpty() && Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this,
                    missingPermissionsList.toArray(new String[missingPermissionsList.size()]),
                    REQUEST_PERMISSION_CODE);
        }
    }

    private void showLoader() {
        openProgressBarView.setVisibility(View.VISIBLE);
    }

    @Override
    public void onClick(View view) {

        if (isDroneConnected) {
            showLoader();
            Intent intent = new Intent(this, VideoActivity.class);
            this.startActivity(intent);
        } else {
            Toast.makeText(getApplicationContext(), R.string.toast_message_connect_drone, Toast.LENGTH_LONG).show();
        }
    }
}

