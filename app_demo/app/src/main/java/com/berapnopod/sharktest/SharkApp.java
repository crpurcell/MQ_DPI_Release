package com.berapnopod.sharktest;

import android.app.Application;
import android.content.Context;

import com.secneo.sdk.Helper;

import org.greenrobot.eventbus.EventBus;

import android.app.Application;
import android.content.Context;

import com.secneo.sdk.Helper;


public class SharkApp extends Application {
    private SDKManager sdkManager;

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
        Helper.install(this);

        if (sdkManager == null) {
            sdkManager = new SDKManager(this, EventBus.getDefault());
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        sdkManager.onCreate();
    }

    public SDKManager getSdkManager() {
        return sdkManager;
    }
}

