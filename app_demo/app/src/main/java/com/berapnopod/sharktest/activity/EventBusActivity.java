package com.berapnopod.sharktest.activity;

import android.support.v7.app.AppCompatActivity;

import org.greenrobot.eventbus.EventBus;

/* https://github.com/greenrobot/EventBus */
abstract class EventBusActivity extends AppCompatActivity {
    @Override
    protected void onStart() {
        super.onStart();
        EventBus.getDefault().register(this);
    }

    @Override
    protected void onStop() {
        super.onStop();
        EventBus.getDefault().unregister(this);
    }
}
