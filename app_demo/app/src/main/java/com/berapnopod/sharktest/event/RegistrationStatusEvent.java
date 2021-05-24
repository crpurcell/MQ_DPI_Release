package com.berapnopod.sharktest.event;

/**
 * DJI SDK registration event
 */
public class RegistrationStatusEvent {

    public final boolean isSDKRegistered;

    public RegistrationStatusEvent(boolean registrationStatus) {
        this.isSDKRegistered = registrationStatus;
    }
}