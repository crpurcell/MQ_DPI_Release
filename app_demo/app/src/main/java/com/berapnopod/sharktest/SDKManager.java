package com.berapnopod.sharktest;

import android.app.Application;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.widget.Toast;

import com.berapnopod.sharktest.event.ConnectivityChangeEvent;
import com.berapnopod.sharkdemo.R;
import com.berapnopod.sharktest.event.RegistrationStatusEvent;

import org.greenrobot.eventbus.EventBus;

import dji.common.error.DJIError;
import dji.common.error.DJISDKError;
import dji.log.DJILog;
import dji.sdk.base.BaseComponent;
import dji.sdk.base.BaseProduct;
import dji.sdk.products.Aircraft;
import dji.sdk.sdkmanager.DJISDKManager;

public class SDKManager {
    private static final String TAG = SDKManager.class.getName();

    private static BaseProduct product;

    private Application application;
    private EventBus eventBus;

    private BaseComponent.ComponentListener mDJIComponentListener;
    private DJISDKManager.SDKManagerCallback mDJISDKManagerCallback;

    SDKManager(Application application, EventBus eventBus) {
        this.application = application;
        this.eventBus = eventBus;
    }

    public void onCreate() {
        mDJIComponentListener = new BaseComponent.ComponentListener() {
            @Override
            public void onConnectivityChange(boolean isConnected) {
                notifyStatusChange();
            }
        };

        /*
         * When starting SDK services, an instance of interface DJISDKManager.DJISDKManagerCallback will be used to listen to
         * the SDK Registration result and the product changing.
         */
        mDJISDKManagerCallback = new DJISDKManager.SDKManagerCallback() {
            @Override
            public void onRegister(DJIError error) {
                if (error == DJISDKError.REGISTRATION_SUCCESS) {
                    DJILog.e("App registration", DJISDKError.REGISTRATION_SUCCESS.getDescription());
                    Log.v(TAG, "registration status is successful:" + error.getDescription());
                    DJISDKManager.getInstance().startConnectionToProduct();
                    notifyRegistrationStatus(true);
                } else {
                    Handler handler = new Handler(Looper.getMainLooper());
                    handler.post(new Runnable() {

                        @Override
                        public void run() {
                            Toast.makeText(application, R.string.sdk_registration_message, Toast.LENGTH_LONG)
                                    .show();
                        }
                    });
                    Log.v(TAG, "registration status:" + error.getDescription());
                    notifyRegistrationStatus(false);
                }

                Log.v(TAG, "registration status general:" + error.getDescription());
            }

            @Override
            public void onProductDisconnect() {
                notifyStatusChange();
            }

            @Override
            public void onProductConnect(BaseProduct baseProduct) {
                notifyStatusChange();
            }

            @Override
            public void onComponentChange(BaseProduct.ComponentKey componentKey, BaseComponent oldComponent, BaseComponent newComponent) {
                if(newComponent != null) {
                    newComponent.setComponentListener(new BaseComponent.ComponentListener() {
                        @Override
                        public void onConnectivityChange(boolean b) {
                            notifyStatusChange();
                        }
                    });
                }
            }
        };

        registerSDK();
    }

    public void registerSDK() {
        int permissionCheck = ContextCompat.checkSelfPermission(application, android.Manifest.permission.READ_PHONE_STATE);
        if (permissionCheck == 0 || Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            DJISDKManager.getInstance().registerApp(application, mDJISDKManagerCallback);
        }
    }

    /**
     * Gets instance of the specific product connected after the
     * API KEY is successfully validated. Please make sure the
     * API_KEY has been added in the Manifest
     */
    public BaseProduct getProductInstance() {
        if (null == product) {
            product = DJISDKManager.getInstance().getProduct();
        }
        return product;
    }

    public Aircraft getAircraftInstance() {
        if (!isAircraftConnected()) return null;
        return (Aircraft) getProductInstance();
    }

    private boolean isAircraftConnected() {
        return getProductInstance() != null && getProductInstance() instanceof Aircraft;
    }

    private void notifyStatusChange() {
        eventBus.post(new ConnectivityChangeEvent());
    }

    // Use EventBus to post the DJI SDK registration status
    private void notifyRegistrationStatus(boolean status){
        eventBus.post(new RegistrationStatusEvent(status));
    }

    // This doesn't really work
    public boolean checkSDKStatus() {
        return DJISDKManager.getInstance().hasSDKRegistered();
    }
}
