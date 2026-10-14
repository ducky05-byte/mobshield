package com.mobileshield.companion;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.VpnService;
import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.provider.Settings;
import android.util.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * MobileShield VPN Service.
 * Intercepts outbound device traffic via TUN interface, streams metadata & raw packet
 * bytes to laptop backend over USB reverse tunnel, and forwards traffic over protected
 * sockets so normal internet browsing remains fully functional.
 */
public class MobileShieldVpnService extends VpnService {

    private static final String TAG = "MobileShieldVPN";
    private static final String CHANNEL_ID = "mobileshield_vpn_channel";
    private static final String LAPTOP_API_URL = "http://127.0.0.1:8000/api/packets";
    private static final int MAX_PENDING_PACKETS = 300;

    public static final String ACTION_START = "com.mobileshield.START";
    public static final String ACTION_STOP = "com.mobileshield.STOP";

    // Public Telemetry Indicators
    public static final AtomicLong totalCaptured = new AtomicLong(0);
    public static final AtomicLong totalSent = new AtomicLong(0);
    private static final AtomicLong totalDroppedBeforeDelivery = new AtomicLong(0);
    public static volatile boolean isBackendConnected = false;
    private static volatile boolean isRunningStatic = false;

    private ParcelFileDescriptor vpnInterface = null;
    private FileOutputStream vpnOutputStream = null;
    private Thread captureThread = null;
    private Thread tcpRelayThread = null;
    private Thread cleanerThread = null;
    private Thread flusherThread = null;

    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final List<PacketParser.ParsedPacket> packetBuffer = Collections.synchronizedList(new ArrayList<>());

    // Active session tables
    private final Map<String, UdpSession> udpSessions = new ConcurrentHashMap<>();
    private final Map<String, TcpSession> tcpSessions = new ConcurrentHashMap<>();

    public static boolean isServiceRunning() {
        return isRunningStatic;
    }

    // --- Inner Session Classes ---

    private class UdpSession {
        final String clientIp;
        final int clientPort;
        final String dstIp;
        final int dstPort;
        final boolean isIpv6;
        DatagramSocket socket;
        volatile long lastActive;
        volatile boolean closed = false;

        UdpSession(String clientIp, int clientPort, String dstIp, int dstPort, boolean isIpv6) {
            this.clientIp = clientIp;
            this.clientPort = clientPort;
            this.dstIp = dstIp;
            this.dstPort = dstPort;
            this.isIpv6 = isIpv6;
            this.lastActive = System.currentTimeMillis();
        }

        void close() {
            closed = true;
            if (socket != null && !socket.isClosed()) {
                try {
                    socket.close();
                } catch (Exception ignored) {}
            }
        }
    }

    private class TcpSession {
        final String clientIp;
        final int clientPort;
        final String dstIp;
        final int dstPort;
        final boolean isIpv6;
        SocketChannel channel;
        long serverSeq;
        long clientAckExpected;
        final List<byte[]> pendingOutbound = new ArrayList<>();
        volatile long lastActive;
        volatile boolean isClosed = false;

        TcpSession(String clientIp, int clientPort, String dstIp, int dstPort, long clientInitialSeq, boolean isIpv6) {
            this.clientIp = clientIp;
            this.clientPort = clientPort;
            this.dstIp = dstIp;
            this.dstPort = dstPort;
            this.isIpv6 = isIpv6;
            this.clientAckExpected = (clientInitialSeq + 1) & 0xFFFFFFFFL;
            this.serverSeq = (long) (Math.random() * 1000000) + 1000;
            this.lastActive = System.currentTimeMillis();
        }

        void close() {
            isClosed = true;
            if (channel != null) {
                try {
                    channel.close();
                } catch (Exception ignored) {}
            }
        }
    }

    // --- Service Lifecycle ---

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            stopVpn();
            stopSelf();
            return START_NOT_STICKY;
        }

        startForegroundServiceNotification();
        startVpn();
        return START_STICKY;
    }

    private void startForegroundServiceNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "MobileShield Traffic Monitor",
                    NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Audits live network packets and streams flow telemetry to laptop.");
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) {
                nm.createNotificationChannel(channel);
            }
        }

        Intent openIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, openIntent,
                PendingIntent.FLAG_IMMUTABLE
        );

        Notification.Builder nb;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nb = new Notification.Builder(this, CHANNEL_ID);
        } else {
            nb = new Notification.Builder(this);
        }

        Notification notification = nb
                .setContentTitle("MobileShield Monitoring Active")
                .setContentText("Capturing packet metadata & streaming to laptop backend.")
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentIntent(pendingIntent)
                .build();

        startForeground(1001, notification);
    }

    private synchronized void startVpn() {
        if (isRunning.get()) {
            return;
        }

        try {
            Builder builder = new Builder();
            builder.setSession("MobileShield Monitor");

            // IPv4 Subnet and default route
            builder.addAddress("10.1.1.2", 24);
            builder.addRoute("0.0.0.0", 0);

            // IPv6 Subnet and default route
            try {
                builder.addAddress("fd00:1::2", 64);
                builder.addRoute("::", 0);
            } catch (Exception e) {
                Log.w(TAG, "IPv6 route setup warning: " + e.getMessage());
            }

            // High-reliability public DNS servers
            builder.addDnsServer("8.8.8.8");
            builder.addDnsServer("1.1.1.1");
            try {
                builder.addDnsServer("2001:4860:4860::8888");
                builder.addDnsServer("2606:4700:4700::1111");
            } catch (Exception ignored) {}

            builder.setMtu(1500);

            // Disallow companion package to exempt reverse-tunnel HTTP telemetry from TUN loop
            try {
                builder.addDisallowedApplication(getPackageName());
            } catch (PackageManager.NameNotFoundException ignored) {}

            vpnInterface = builder.establish();
            if (vpnInterface == null) {
                Log.e(TAG, "Failed to establish VPN interface.");
                broadcastTelemetry();
                return;
            }

            vpnOutputStream = new FileOutputStream(vpnInterface.getFileDescriptor());
            isRunning.set(true);
            isRunningStatic = true;

            // Start capture loop
            captureThread = new Thread(this::runCaptureLoop, "MobileShield-CaptureThread");
            captureThread.start();

            // Start TCP relay loop
            tcpRelayThread = new Thread(this::runTcpRelayLoop, "MobileShield-TcpRelayThread");
            tcpRelayThread.start();

            // Start session cleaner loop
            cleanerThread = new Thread(this::runSessionCleanerLoop, "MobileShield-CleanerThread");
            cleanerThread.start();

            // Start periodic flusher thread
            flusherThread = new Thread(this::runBufferFlusherLoop, "MobileShield-FlusherThread");
            flusherThread.start();

            Log.i(TAG, "MobileShield VpnService successfully started. TUN interface is active.");
            broadcastTelemetry();
        } catch (Exception e) {
            Log.e(TAG, "Exception starting VPN: " + e.getMessage(), e);
            broadcastTelemetry();
        }
    }

    public synchronized void writeToVpn(byte[] packetBytes) {
        if (!isRunning.get() || vpnOutputStream == null) return;
        try {
            vpnOutputStream.write(packetBytes);
            // Record return packet for telemetry & PCAP write
            PacketParser.ParsedPacket pkt = PacketParser.parse(packetBytes, packetBytes.length);
            if (pkt.isValid) {
                totalCaptured.incrementAndGet();
                enqueuePacket(pkt);
            }
        } catch (Exception e) {
            Log.w(TAG, "Error injecting return packet to TUN: " + e.getMessage());
        }
    }

    // --- Packet Capture & Dispatch Loop ---

    private void runCaptureLoop() {
        FileInputStream in = new FileInputStream(vpnInterface.getFileDescriptor());
        byte[] buffer = new byte[32767];

        Log.i(TAG, "Capture loop entering active read loop on TUN fd.");

        while (isRunning.get()) {
            try {
                int length = in.read(buffer);
                if (length > 0) {
                    byte[] packetCopy = Arrays.copyOf(buffer, length);

                    // 1. Telemetry ingestion
                    PacketParser.ParsedPacket pkt = PacketParser.parse(packetCopy, length);
                    if (pkt.isValid) {
                        long cap = totalCaptured.incrementAndGet();
                        enqueuePacket(pkt);
                        if (cap == 1 || cap % 25 == 0) {
                            Log.i(TAG, String.format(Locale.US, "Captured packet #%d: proto=%d, len=%d, %s:%d -> %s:%d",
                                    cap, pkt.protocol, pkt.packetLen, pkt.srcIp, pkt.srcPort, pkt.dstIp, pkt.dstPort));
                        }
                    }

                    // 2. Real forward dispatching
                    dispatchOutboundPacket(packetCopy, length);
                }
            } catch (Exception e) {
                if (isRunning.get()) {
                    Log.w(TAG, "Read error in capture loop: " + e.getMessage());
                }
                break;
            }
        }
    }

    private void runBufferFlusherLoop() {
        while (isRunning.get()) {
            try {
                if (packetBuffer.isEmpty()) {
                    Thread.sleep(200);
                } else if (!flushPacketBuffer()) {
                    Thread.sleep(1000);
                }
            } catch (InterruptedException e) {
                break;
            }
        }
    }

    private void enqueuePacket(PacketParser.ParsedPacket packet) {
        synchronized (packetBuffer) {
            if (packetBuffer.size() >= MAX_PENDING_PACKETS) {
                long dropped = totalDroppedBeforeDelivery.incrementAndGet();
                if (dropped == 1 || dropped % 25 == 0) {
                    Log.e(TAG, "Pending delivery queue full; dropped " + dropped
                            + " captured packet(s) before delivery.");
                }
                return;
            }
            packetBuffer.add(packet);
        }
    }

    private void dispatchOutboundPacket(byte[] packet, int length) {
        if (length < 20) return;
        int version = (packet[0] >> 4) & 0x0F;

        if (version == 4) {
            int ihl = (packet[0] & 0x0F) * 4;
            int protocol = packet[9] & 0xFF;

            String srcIp = String.format(Locale.US, "%d.%d.%d.%d",
                    packet[12] & 0xFF, packet[13] & 0xFF, packet[14] & 0xFF, packet[15] & 0xFF);
            String dstIp = String.format(Locale.US, "%d.%d.%d.%d",
                    packet[16] & 0xFF, packet[17] & 0xFF, packet[18] & 0xFF, packet[19] & 0xFF);

            if (protocol == 17 && length >= ihl + 8) { // UDP
                handleOutboundUdp(packet, length, ihl, srcIp, dstIp, false);
            } else if (protocol == 6 && length >= ihl + 20) { // TCP
                handleOutboundTcp(packet, length, ihl, srcIp, dstIp, false);
            }
        } else if (version == 6 && length >= 40) {
            int protocol = packet[6] & 0xFF;
            StringBuilder srcBuilder = new StringBuilder();
            StringBuilder dstBuilder = new StringBuilder();
            for (int i = 8; i < 24; i += 2) {
                if (srcBuilder.length() > 0) srcBuilder.append(":");
                srcBuilder.append(String.format("%x", ((packet[i] & 0xFF) << 8) | (packet[i + 1] & 0xFF)));
            }
            for (int i = 24; i < 40; i += 2) {
                if (dstBuilder.length() > 0) dstBuilder.append(":");
                dstBuilder.append(String.format("%x", ((packet[i] & 0xFF) << 8) | (packet[i + 1] & 0xFF)));
            }
            String srcIp = srcBuilder.toString();
            String dstIp = dstBuilder.toString();

            if (protocol == 17 && length >= 48) { // UDP
                handleOutboundUdp(packet, length, 40, srcIp, dstIp, true);
            } else if (protocol == 6 && length >= 60) { // TCP
                handleOutboundTcp(packet, length, 40, srcIp, dstIp, true);
            }
        }
    }

    // --- UDP Forwarding Implementation ---

    private void handleOutboundUdp(byte[] packet, int length, int ihl, String srcIp, String dstIp, boolean isIpv6) {
        int srcPort = ((packet[ihl] & 0xFF) << 8) | (packet[ihl + 1] & 0xFF);
        int dstPort = ((packet[ihl + 2] & 0xFF) << 8) | (packet[ihl + 3] & 0xFF);
        int udpLen = ((packet[ihl + 4] & 0xFF) << 8) | (packet[ihl + 5] & 0xFF);
        int payloadLen = Math.min(udpLen - 8, length - ihl - 8);

        byte[] payload = null;
        if (payloadLen > 0) {
            payload = new byte[payloadLen];
            System.arraycopy(packet, ihl + 8, payload, 0, payloadLen);
        }

        String sessionKey = srcPort + "->" + dstIp + ":" + dstPort;
        UdpSession session = udpSessions.get(sessionKey);
        if (session == null) {
            session = new UdpSession(srcIp, srcPort, dstIp, dstPort, isIpv6);
            try {
                session.socket = new DatagramSocket();
                protect(session.socket); // CRITICAL: bypass VPN for real outbound transmission
                udpSessions.put(sessionKey, session);

                final UdpSession activeSession = session;
                new Thread(() -> runUdpReceiver(activeSession), "UdpReceiver-" + sessionKey).start();
            } catch (Exception e) {
                Log.w(TAG, "Failed creating UDP socket for " + sessionKey + ": " + e.getMessage());
                return;
            }
        }

        session.lastActive = System.currentTimeMillis();
        if (payload != null && session.socket != null && !session.socket.isClosed()) {
            final byte[] sendPayload = payload;
            final UdpSession currentSession = session;
            new Thread(() -> {
                try {
                    InetAddress destAddr = InetAddress.getByName(dstIp);
                    DatagramPacket outPkt = new DatagramPacket(sendPayload, sendPayload.length, destAddr, dstPort);
                    currentSession.socket.send(outPkt);
                } catch (Exception e) {
                    Log.w(TAG, "UDP send error: " + e.getMessage());
                }
            }).start();
        }
    }

    private void runUdpReceiver(UdpSession session) {
        byte[] recvBuf = new byte[4096];
        DatagramPacket recvPkt = new DatagramPacket(recvBuf, recvBuf.length);

        while (isRunning.get() && !session.closed) {
            try {
                session.socket.setSoTimeout(15000); // 15-second idle timeout
                session.socket.receive(recvPkt);

                session.lastActive = System.currentTimeMillis();
                byte[] respPayload = Arrays.copyOf(recvPkt.getData(), recvPkt.getLength());

                byte[] returnPkt;
                if (session.isIpv6) {
                    returnPkt = PacketCrafter.buildUdpPacketV6(
                            session.dstIp, session.clientIp, session.dstPort, session.clientPort, respPayload
                    );
                } else {
                    returnPkt = PacketCrafter.buildUdpPacket(
                            session.dstIp, session.clientIp, session.dstPort, session.clientPort, respPayload
                    );
                }
                writeToVpn(returnPkt);
            } catch (Exception e) {
                break;
            }
        }
        session.close();
        udpSessions.values().remove(session);
    }

    // --- TCP Forwarding Implementation ---

    private void handleOutboundTcp(byte[] packet, int length, int ihl, String srcIp, String dstIp, boolean isIpv6) {
        int srcPort = ((packet[ihl] & 0xFF) << 8) | (packet[ihl + 1] & 0xFF);
        int dstPort = ((packet[ihl + 2] & 0xFF) << 8) | (packet[ihl + 3] & 0xFF);
        long seq = (((long) packet[ihl + 4] & 0xFF) << 24)
                | (((long) packet[ihl + 5] & 0xFF) << 16)
                | (((long) packet[ihl + 6] & 0xFF) << 8)
                | (((long) packet[ihl + 7] & 0xFF));
        int tcpHdrLen = ((packet[ihl + 12] >> 4) & 0x0F) * 4;
        int flags = packet[ihl + 13] & 0xFF;
        int payloadLen = Math.max(length - ihl - tcpHdrLen, 0);

        String sessionKey = srcPort + "->" + dstIp + ":" + dstPort;
        boolean isSyn = (flags & 0x02) != 0;
        boolean isAck = (flags & 0x10) != 0;
        boolean isFin = (flags & 0x01) != 0;
        boolean isRst = (flags & 0x04) != 0;

        if (isSyn && !isAck) {
            TcpSession session = new TcpSession(srcIp, srcPort, dstIp, dstPort, seq, isIpv6);
            try {
                session.channel = SocketChannel.open();
                session.channel.configureBlocking(false);
                protect(session.channel.socket()); // CRITICAL: bypass VPN
                session.channel.connect(new InetSocketAddress(dstIp, dstPort));

                tcpSessions.put(sessionKey, session);

                byte[] synAck;
                if (isIpv6) {
                    synAck = PacketCrafter.buildTcpPacketV6(
                            dstIp, srcIp, dstPort, srcPort,
                            session.serverSeq, session.clientAckExpected,
                            0x12, 65535, null
                    );
                } else {
                    synAck = PacketCrafter.buildTcpPacket(
                            dstIp, srcIp, dstPort, srcPort,
                            session.serverSeq, session.clientAckExpected,
                            0x12, 65535, null
                    );
                }
                writeToVpn(synAck);
                session.serverSeq++;
            } catch (Exception e) {
                Log.w(TAG, "TCP connection initiation failed for " + sessionKey + ": " + e.getMessage());
                byte[] rst = isIpv6
                        ? PacketCrafter.buildTcpRstV6(dstIp, srcIp, dstPort, srcPort, 0, (seq + 1) & 0xFFFFFFFFL)
                        : PacketCrafter.buildTcpRst(dstIp, srcIp, dstPort, srcPort, 0, (seq + 1) & 0xFFFFFFFFL);
                writeToVpn(rst);
            }
            return;
        }

        TcpSession session = tcpSessions.get(sessionKey);
        if (session == null) {
            if (!isRst) {
                byte[] rst = isIpv6
                        ? PacketCrafter.buildTcpRstV6(dstIp, srcIp, dstPort, srcPort, 0, (seq + 1) & 0xFFFFFFFFL)
                        : PacketCrafter.buildTcpRst(dstIp, srcIp, dstPort, srcPort, 0, (seq + 1) & 0xFFFFFFFFL);
                writeToVpn(rst);
            }
            return;
        }

        session.lastActive = System.currentTimeMillis();

        if (isRst) {
            session.close();
            tcpSessions.remove(sessionKey);
            return;
        }

        if (isFin) {
            session.clientAckExpected = (seq + 1) & 0xFFFFFFFFL;
            byte[] finAck = isIpv6
                    ? PacketCrafter.buildTcpPacketV6(dstIp, srcIp, dstPort, srcPort, session.serverSeq, session.clientAckExpected, 0x11, 65535, null)
                    : PacketCrafter.buildTcpPacket(dstIp, srcIp, dstPort, srcPort, session.serverSeq, session.clientAckExpected, 0x11, 65535, null);
            writeToVpn(finAck);
            session.close();
            tcpSessions.remove(sessionKey);
            return;
        }

        // Data payload from client (e.g. TLS ClientHello, HTTP request)
        if (payloadLen > 0) {
            byte[] payload = new byte[payloadLen];
            System.arraycopy(packet, ihl + tcpHdrLen, payload, 0, payloadLen);
            session.clientAckExpected = (seq + payloadLen) & 0xFFFFFFFFL;

            // Immediately ACK received client data
            byte[] ackPkt = isIpv6
                    ? PacketCrafter.buildTcpPacketV6(dstIp, srcIp, dstPort, srcPort, session.serverSeq, session.clientAckExpected, 0x10, 65535, null)
                    : PacketCrafter.buildTcpPacket(dstIp, srcIp, dstPort, srcPort, session.serverSeq, session.clientAckExpected, 0x10, 65535, null);
            writeToVpn(ackPkt);

            // Forward to real server
            synchronized (session.pendingOutbound) {
                session.pendingOutbound.add(payload);
            }
        }
    }

    private void runTcpRelayLoop() {
        ByteBuffer readBuf = ByteBuffer.allocate(16384);

        while (isRunning.get()) {
            try {
                if (tcpSessions.isEmpty()) {
                    Thread.sleep(10);
                    continue;
                }

                for (Iterator<Map.Entry<String, TcpSession>> it = tcpSessions.entrySet().iterator(); it.hasNext(); ) {
                    Map.Entry<String, TcpSession> entry = it.next();
                    TcpSession session = entry.getValue();

                    if (session.isClosed || session.channel == null) {
                        it.remove();
                        continue;
                    }

                    try {
                        if (session.channel.isConnectionPending()) {
                            session.channel.finishConnect();
                        }

                        if (session.channel.isConnected()) {
                            // 1. Flush any pending outbound data from client
                            synchronized (session.pendingOutbound) {
                                while (!session.pendingOutbound.isEmpty()) {
                                    byte[] data = session.pendingOutbound.get(0);
                                    ByteBuffer buf = ByteBuffer.wrap(data);
                                    session.channel.write(buf);
                                    if (!buf.hasRemaining()) {
                                        session.pendingOutbound.remove(0);
                                    } else {
                                        break;
                                    }
                                }
                            }

                            // 2. Read incoming server data and forward into TUN
                            readBuf.clear();
                            int bytesRead = session.channel.read(readBuf);
                            if (bytesRead > 0) {
                                session.lastActive = System.currentTimeMillis();
                                readBuf.flip();
                                byte[] serverData = new byte[bytesRead];
                                readBuf.get(serverData);

                                int mss = 1360;
                                int offset = 0;
                                while (offset < bytesRead) {
                                    int sliceLen = Math.min(mss, bytesRead - offset);
                                    byte[] slice = new byte[sliceLen];
                                    System.arraycopy(serverData, offset, slice, 0, sliceLen);

                                    byte[] tunPkt;
                                    if (session.isIpv6) {
                                        tunPkt = PacketCrafter.buildTcpPacketV6(
                                                session.dstIp, session.clientIp,
                                                session.dstPort, session.clientPort,
                                                session.serverSeq, session.clientAckExpected,
                                                0x18, 65535, slice
                                        );
                                    } else {
                                        tunPkt = PacketCrafter.buildTcpPacket(
                                                session.dstIp, session.clientIp,
                                                session.dstPort, session.clientPort,
                                                session.serverSeq, session.clientAckExpected,
                                                0x18, 65535, slice
                                        );
                                    }
                                    writeToVpn(tunPkt);
                                    session.serverSeq = (session.serverSeq + sliceLen) & 0xFFFFFFFFL;
                                    offset += sliceLen;
                                }
                            } else if (bytesRead == -1) {
                                byte[] finPkt = session.isIpv6
                                        ? PacketCrafter.buildTcpPacketV6(session.dstIp, session.clientIp, session.dstPort, session.clientPort, session.serverSeq, session.clientAckExpected, 0x11, 65535, null)
                                        : PacketCrafter.buildTcpPacket(session.dstIp, session.clientIp, session.dstPort, session.clientPort, session.serverSeq, session.clientAckExpected, 0x11, 65535, null);
                                writeToVpn(finPkt);
                                session.close();
                                it.remove();
                            }
                        }
                    } catch (Exception e) {
                        session.close();
                        it.remove();
                    }
                }

                Thread.sleep(5);
            } catch (Exception ignored) {}
        }
    }

    private void runSessionCleanerLoop() {
        while (isRunning.get()) {
            try {
                Thread.sleep(10000);
                long now = System.currentTimeMillis();

                // Clean stale UDP sessions (> 30s)
                for (Iterator<Map.Entry<String, UdpSession>> it = udpSessions.entrySet().iterator(); it.hasNext(); ) {
                    UdpSession s = it.next().getValue();
                    if (now - s.lastActive > 30000) {
                        s.close();
                        it.remove();
                    }
                }

                // Clean stale TCP sessions (> 45s)
                for (Iterator<Map.Entry<String, TcpSession>> it = tcpSessions.entrySet().iterator(); it.hasNext(); ) {
                    TcpSession s = it.next().getValue();
                    if (now - s.lastActive > 45000) {
                        s.close();
                        it.remove();
                    }
                }
            } catch (Exception ignored) {}
        }
    }

    // --- Packet Streaming to Laptop API ---

    private boolean flushPacketBuffer() {
        List<PacketParser.ParsedPacket> batch;
        synchronized (packetBuffer) {
            if (packetBuffer.isEmpty()) return true;
            int take = Math.min(packetBuffer.size(), 50);
            batch = new ArrayList<>(packetBuffer.subList(0, take));
            packetBuffer.subList(0, take).clear();
        }

        HttpURLConnection conn = null;
        boolean success = false;
        try {
            String authToken = DeviceCredentials.getToken(this);
            String deviceId = Settings.Secure.getString(
                    getContentResolver(), Settings.Secure.ANDROID_ID
            );
            if (authToken == null || deviceId == null) {
                throw new IllegalStateException(
                        "Device enrollment is required before packet submission."
                );
            }
            URL url = new URL(LAPTOP_API_URL);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("X-MobileShield-Token", authToken);
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(15000);
            conn.setDoOutput(true);

            StringBuilder json = new StringBuilder();
            json.append("{\"device_id\":\"").append(deviceId).append("\",\"packets\":[");
            for (int i = 0; i < batch.size(); i++) {
                PacketParser.ParsedPacket p = batch.get(i);
                if (i > 0) json.append(",");
                json.append("{")
                    .append("\"timestamp\":").append(p.timestamp).append(",")
                    .append("\"src_ip\":\"").append(p.srcIp != null ? p.srcIp : "").append("\",")
                    .append("\"dst_ip\":\"").append(p.dstIp != null ? p.dstIp : "").append("\",")
                    .append("\"src_port\":").append(p.srcPort).append(",")
                    .append("\"dst_port\":").append(p.dstPort).append(",")
                    .append("\"protocol\":").append(p.protocol).append(",")
                    .append("\"packet_len\":").append(p.packetLen).append(",")
                    .append("\"tcp_flags\":").append(p.tcpFlags).append(",")
                    .append("\"ttl\":").append(p.ttl).append(",")
                    .append("\"raw_packet_hex\":\"").append(p.rawHex != null ? p.rawHex : "").append("\"")
                    .append("}");
            }
            json.append("]}");

            byte[] postData = json.toString().getBytes(StandardCharsets.UTF_8);
            conn.setFixedLengthStreamingMode(postData.length);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(postData);
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                ByteArrayOutputStream responseBody = new ByteArrayOutputStream();
                try (InputStream input = conn.getInputStream()) {
                    byte[] responseBuffer = new byte[1024];
                    int bytesRead;
                    while ((bytesRead = input.read(responseBuffer)) != -1) {
                        responseBody.write(responseBuffer, 0, bytesRead);
                    }
                }
                JSONObject response = new JSONObject(
                        new String(responseBody.toByteArray(), StandardCharsets.UTF_8)
                );
                success = "SUCCESS".equals(response.optString("status"))
                        && response.optInt("packets_processed", -1) == batch.size();
                if (!success) {
                    Log.e(TAG, "Backend accepted HTTP request but reported an incomplete packet batch.");
                }
            } else {
                Log.e(TAG, "Laptop backend responded with HTTP " + responseCode);
            }
        } catch (Exception e) {
            Log.e(TAG, "Packet streaming delivery failed; batch will be retried.", e);
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }

        if (success) {
            long total = totalSent.addAndGet(batch.size());
            isBackendConnected = true;
            Log.i(TAG, "Delivered " + batch.size() + " packets to laptop. Total sent: " + total);
            broadcastTelemetry();
            return true;
        }

        isBackendConnected = false;
        synchronized (packetBuffer) {
            int overflow = packetBuffer.size() + batch.size() - MAX_PENDING_PACKETS;
            for (int i = 0; i < overflow && !packetBuffer.isEmpty(); i++) {
                packetBuffer.remove(packetBuffer.size() - 1);
            }
            if (overflow > 0) {
                long dropped = totalDroppedBeforeDelivery.addAndGet(overflow);
                Log.e(TAG, "Delivery retry queue full; dropped " + overflow
                        + " newest packet(s). Total dropped before delivery: " + dropped);
            }
            packetBuffer.addAll(0, batch);
        }
        broadcastTelemetry();
        return false;
    }

    private void broadcastTelemetry() {
        Intent intent = new Intent(MainActivity.ACTION_TELEMETRY);
        intent.putExtra(MainActivity.EXTRA_CAPTURED, totalCaptured.get());
        intent.putExtra(MainActivity.EXTRA_SENT, totalSent.get());
        intent.putExtra(MainActivity.EXTRA_CONNECTED, isBackendConnected);

        String status;
        if (!isRunning.get()) {
            status = "Idle (Capture Stopped)";
        } else if (isBackendConnected) {
            status = "ACTIVE (Streaming to Laptop)";
        } else {
            status = "CAPTURING (Connecting to 127.0.0.1:8000...)";
        }
        intent.putExtra(MainActivity.EXTRA_STATUS, status);

        sendBroadcast(intent);
    }

    // --- Shutdown & Resource Cleanup ---

    private synchronized void stopVpn() {
        isRunning.set(false);
        isRunningStatic = false;

        if (captureThread != null) {
            captureThread.interrupt();
            captureThread = null;
        }
        if (tcpRelayThread != null) {
            tcpRelayThread.interrupt();
            tcpRelayThread = null;
        }
        if (cleanerThread != null) {
            cleanerThread.interrupt();
            cleanerThread = null;
        }
        if (flusherThread != null) {
            flusherThread.interrupt();
            flusherThread = null;
        }

        // Close all active UDP sockets
        for (UdpSession s : udpSessions.values()) {
            s.close();
        }
        udpSessions.clear();

        // Close all active TCP channels
        for (TcpSession s : tcpSessions.values()) {
            s.close();
        }
        tcpSessions.clear();

        if (vpnOutputStream != null) {
            try {
                vpnOutputStream.close();
            } catch (Exception ignored) {}
            vpnOutputStream = null;
        }

        if (vpnInterface != null) {
            try {
                vpnInterface.close();
            } catch (Exception ignored) {}
            vpnInterface = null;
        }

        stopForeground(true);
        Log.i(TAG, "MobileShield VpnService stopped.");
        broadcastTelemetry();
    }

    @Override
    public void onDestroy() {
        stopVpn();
        super.onDestroy();
    }
}
