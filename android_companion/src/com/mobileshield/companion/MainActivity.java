package com.mobileshield.companion;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.VpnService;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {

    private static final int VPN_REQUEST_CODE = 0x101;
    public static final String ACTION_TELEMETRY = "com.mobileshield.companion.TELEMETRY_UPDATE";
    public static final String EXTRA_CAPTURED = "captured";
    public static final String EXTRA_SENT = "sent";
    public static final String EXTRA_STATUS = "status_text";
    public static final String EXTRA_CONNECTED = "is_connected";

    private Button btnToggle;
    private Button btnEnroll;
    private EditText etPairingCode;
    private TextView tvStatus;
    private TextView tvPacketStats;
    private boolean isMonitoring = false;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private final Runnable uiUpdater = new Runnable() {
        @Override
        public void run() {
            updateUiState();
            uiHandler.postDelayed(this, 1000);
        }
    };

    private final BroadcastReceiver telemetryReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent != null && ACTION_TELEMETRY.equals(intent.getAction())) {
                long captured = intent.getLongExtra(EXTRA_CAPTURED, 0);
                long sent = intent.getLongExtra(EXTRA_SENT, 0);
                String statusText = intent.getStringExtra(EXTRA_STATUS);
                boolean isConnected = intent.getBooleanExtra(EXTRA_CONNECTED, false);

                tvPacketStats.setText(String.format("Packets Captured: %d | Sent to Laptop: %d", captured, sent));
                if (statusText != null && !statusText.isEmpty()) {
                    tvStatus.setText("Status: " + statusText);
                    tvStatus.setTextColor(isConnected ? 0xFF22C55E : 0xFFFBBF24); // Green vs Amber
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnToggle = findViewById(R.id.btnToggleVpn);
        btnEnroll = findViewById(R.id.btnEnrollDevice);
        etPairingCode = findViewById(R.id.etPairingCode);
        tvStatus = findViewById(R.id.tvStatus);
        tvPacketStats = findViewById(R.id.tvPacketStats);

        btnEnroll.setOnClickListener(v -> enrollDevice());
        btnToggle.setOnClickListener(v -> {
            if (!isMonitoring) {
                requestAndStartVpn();
            } else {
                stopMonitoring();
            }
        });

        updateUiState();
    }

    @Override
    protected void onResume() {
        super.onResume();
        IntentFilter filter = new IntentFilter(ACTION_TELEMETRY);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(telemetryReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(telemetryReceiver, filter);
        }
        uiHandler.post(uiUpdater);
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
            unregisterReceiver(telemetryReceiver);
        } catch (Exception ignored) {}
        uiHandler.removeCallbacks(uiUpdater);
    }

    private void updateUiState() {
        boolean active = MobileShieldVpnService.isServiceRunning();
        isMonitoring = active;
        long captured = MobileShieldVpnService.totalCaptured.get();
        long sent = MobileShieldVpnService.totalSent.get();
        boolean connected = MobileShieldVpnService.isBackendConnected;

        tvPacketStats.setText(String.format("Packets Captured: %d | Sent to Laptop: %d", captured, sent));

        if (active) {
            btnToggle.setText("STOP CAPTURE");
            btnToggle.setBackgroundColor(0xFFDC2626); // Red
            if (connected) {
                tvStatus.setText("Status: ACTIVE (Streaming to Laptop)");
                tvStatus.setTextColor(0xFF22C55E); // Green
            } else {
                tvStatus.setText("Status: CAPTURING (Connecting to 127.0.0.1:8000...)");
                tvStatus.setTextColor(0xFFFBBF24); // Amber
            }
        } else {
            btnToggle.setText("START CAPTURE");
            btnToggle.setBackgroundColor(0xFF0284C7); // Blue
            tvStatus.setText("Status: Idle (Capture Stopped)");
            tvStatus.setTextColor(0xFFF1F5F9);
        }
    }

    private void requestAndStartVpn() {
        Intent prepareIntent = VpnService.prepare(this);
        if (prepareIntent != null) {
            startActivityForResult(prepareIntent, VPN_REQUEST_CODE);
        } else {
            onActivityResult(VPN_REQUEST_CODE, RESULT_OK, null);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == VPN_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                startMonitoringService();
            } else {
                Toast.makeText(this, "VPN permission required for MobileShield monitoring", Toast.LENGTH_LONG).show();
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void startMonitoringService() {
        if (!hasEnrolledCredential()) {
            Toast.makeText(this, "Enroll this device before starting capture", Toast.LENGTH_LONG).show();
            return;
        }
        Intent intent = new Intent(this, MobileShieldVpnService.class);
        intent.setAction(MobileShieldVpnService.ACTION_START);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        isMonitoring = true;
        btnToggle.setText("STOP CAPTURE");
        btnToggle.setBackgroundColor(0xFFDC2626);
        tvStatus.setText("Status: INITIALIZING CAPTURE...");
        tvStatus.setTextColor(0xFF38BDF8);
        Toast.makeText(this, "MobileShield capture started", Toast.LENGTH_SHORT).show();
    }

    private void stopMonitoring() {
        Intent intent = new Intent(this, MobileShieldVpnService.class);
        intent.setAction(MobileShieldVpnService.ACTION_STOP);
        startService(intent);

        isMonitoring = false;
        btnToggle.setText("START CAPTURE");
        btnToggle.setBackgroundColor(0xFF0284C7);
        tvStatus.setText("Status: Idle (Capture Stopped)");
        tvStatus.setTextColor(0xFFF1F5F9);
        Toast.makeText(this, "MobileShield capture stopped", Toast.LENGTH_SHORT).show();
    }

    private void enrollDevice() {
        String pairingCode = etPairingCode.getText().toString().trim();
        String deviceId = Settings.Secure.getString(
                getContentResolver(), Settings.Secure.ANDROID_ID
        );
        if (pairingCode.isEmpty() || deviceId == null || deviceId.isEmpty()) {
            Toast.makeText(this, "Enter the operator pairing code", Toast.LENGTH_LONG).show();
            return;
        }

        btnEnroll.setEnabled(false);
        new Thread(() -> {
            HttpURLConnection connection = null;
            try {
                URL url = new URL("http://127.0.0.1:8000/api/device/enroll");
                connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json");
                connection.setConnectTimeout(5000);
                connection.setReadTimeout(5000);
                connection.setDoOutput(true);

                JSONObject request = new JSONObject();
                request.put("device_id", deviceId);
                request.put("pairing_code", pairingCode);
                byte[] body = request.toString().getBytes(StandardCharsets.UTF_8);
                try (OutputStream output = connection.getOutputStream()) {
                    output.write(body);
                }

                int responseCode = connection.getResponseCode();
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    throw new IllegalStateException("Enrollment failed with HTTP " + responseCode);
                }
                ByteArrayOutputStream response = new ByteArrayOutputStream();
                try (InputStream input = connection.getInputStream()) {
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = input.read(buffer)) != -1) {
                        response.write(buffer, 0, read);
                    }
                }
                String token = new JSONObject(
                        new String(response.toByteArray(), StandardCharsets.UTF_8)
                ).getString("token");
                DeviceCredentials.saveToken(this, token);

                runOnUiThread(() -> {
                    etPairingCode.setText("");
                    tvStatus.setText("Status: Device enrolled");
                    Toast.makeText(this, "Device enrolled", Toast.LENGTH_SHORT).show();
                });
            } catch (Exception exc) {
                Log.e("MobileShield", "Device enrollment failed", exc);
                runOnUiThread(() -> Toast.makeText(
                        this, "Enrollment failed. Check USB tunnel and pairing code.",
                        Toast.LENGTH_LONG
                ).show());
            } finally {
                if (connection != null) {
                    connection.disconnect();
                }
                runOnUiThread(() -> btnEnroll.setEnabled(true));
            }
        }, "DeviceEnrollmentThread").start();
    }

    private boolean hasEnrolledCredential() {
        try {
            return DeviceCredentials.getToken(this) != null;
        } catch (Exception exc) {
            Log.e("MobileShield", "Unable to read the device credential", exc);
            return false;
        }
    }
}
