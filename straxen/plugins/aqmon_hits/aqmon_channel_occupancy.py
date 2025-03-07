import straxen
from straxen.plugins.aqmon_hits.aqmon_hits import AqmonChannels


class AqMonChannelOccupancy:
    """For V1495 Firmware v10, from Feb. '25 Determines the meaning of the Acquisition Monitor
    Channels originating from the V1495 depending on its config in the DAQ. These are start and stop
    signals of veto- or other intervals. The channels have a name according to its primary usage,

    but the meaning can differ.
      * 'Busy' channels always only provide the busy veto
      * 'Neutron Generator' (NG) channels can provide
                            a) LED-trigger
                            b) Neutron Generator trigger/indicator
                            c) nothing
                            d) 'busy_he' for backwards compatibilty (older runs)
      * 'High Energy Veto' (HEV) channels can provide
                            a) HEV
                            b) HEV-tag
                            c) Fractional Lifetime veto
                            d) Anti Veto

    HEV-tag means, that we have the information of the HEV, whether it makes a veto decision or not,
    but we DO NOT veto the system.
    More info:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:robingb:v1495_firmware_upgrade_v10

    """

    neutron_generator_channel_options = {
        1: "LED_trg",
        2: "neutron_generator",
        3: "busy_he",  # backwards compatibilty
        4: "empty",
    }

    hev_channel_options = {1: "anti_veto", 2: "fractional_lifetime", 3: "hev", 4: "hev_tag"}

    def __init__(self, run_id: int):
        self.run_id = run_id
        self.get_settings()
        self.make_channel_map()

    def get_settings(self):
        self.get_v1495_config()
        # self.get_fake_config()
        self.extract_settings()
        self._settings_check_plausilble()

    def get_v1495_config(self):
        """Query the RunDB to extract V1495 options from metadata of specified run."""
        query = {"number": int(self.run_id)}
        rundb = straxen.utilix.xent_collection()
        doc = rundb.find_one(query)
        v1495_options = doc["daq_config"]["V1495"]["tpc"]
        # turn settings with boolean meaning to actual bools for easier handling
        self.config = {
            k: (bool(v) if k.startswith(("is_", "_use_")) else v) for k, v in v1495_options.items()
        }

    def get_fake_config(self):
        """For testing."""
        print("Using fake config")
        self.config = {
            "is_hev_on": True,
            "is_frac_lt_mode_on": False,
            "is_led_start_stop_activ": False,
            "is_anti_veto_active": True,
            "anti_veto_delay_us": 1,
            "anti_veto_duration_us": 2,
            "fractional_lifetime_veto_on_us": 0,
            "fractional_lifetime_veto_off_us": 0,
            "_use_legacy_port_hev": True,
            "_use_regular_port_trg": False,
            "_use_legacy_port_trg": False,
            "_use_NG_input": True,
            "firmware_version": 10,
        }

    def extract_settings(self):
        """Decode V1495 config."""
        self.firmware_version = self.config.get("firmware_version", 9)

        self.use_ng_input = self.config.get("_use_NG_input", False)
        self.use_led_trg = (
            self.config.get("_use_regular_port_trg", False)
            or self.config.get("_use_legacy_port_trg", False)
        ) and self.config.get("is_led_start_stop_activ", False)

        self.fracLT_on = self.config.get("is_frac_lt_mode_on", False)
        self.fracLT_veto_on_period = self.config.get("fractional_lifetime_veto_on_us", 0)
        self.fracLT_veto_off_period = self.config.get("fractional_lifetime_veto_off_us", 0)

        self.anti_veto_on = self.config.get("is_anti_veto_active", False) and self.use_ng_input
        self.anti_veto_delay_µs = self.config.get("anti_veto_delay_us", 0)
        self.anti_veto_duration_µs = self.config.get("anti_veto_duration_us", 0)

        self.hev_on = self.config.get("is_hev_on", False)

    def _settings_check_plausilble(self):
        """Check if V1495 config makes sense.

        Throw error otherwise. Redax should not allow most of these, but better be safe than sorry.

        """
        if self.fracLT_on:
            if not ((self.fracLT_veto_on_period > 0) and (self.fracLT_veto_off_period > 0)):
                raise RuntimeError(
                    "Fractional Lifetime Veto is set to on, but at least one of its periods is 0."
                )
        if self.anti_veto_on:
            if not self.anti_veto_duration_µs > 0:
                raise RuntimeError("Anti Veto is set to on, but its duration is 0.")
        if self.use_ng_input and self.use_led_trg:
            raise RuntimeError(
                "Neutron generator and LED Trigger are on. This should not be allowed"
            )
        if self.fracLT_on and self.hev_on:
            raise RuntimeError(
                "Fractional Lifetime mode and HEV are on. This should not be allowed."
            )

    def _infer_ng_channel_meaning(self):
        """Infer what output it to be expected on the 'neutron generator' channel."""
        ng_options_reverse = {v: k for k, v in self.neutron_generator_channel_options.items()}

        if self.firmware_version == 9:
            return ng_options_reverse["busy_he"]

        if self.use_ng_input:
            ng_option = ng_options_reverse["neutron_generator"]
        if self.use_led_trg:
            ng_option = ng_options_reverse["LED_trg"]
        if not (self.use_ng_input or self.use_led_trg):
            ng_option = ng_options_reverse["empty"]

        return ng_option

    def _infer_hev_channel_meaning(self):
        """Infer what output it to be expected on the 'high energy veto' channel."""
        hev_options_reverse = {v: k for k, v in self.hev_channel_options.items()}

        if self.firmware_version == 9:
            return hev_options_reverse["hev"]

        if self.fracLT_on:
            hev_option = hev_options_reverse["fractional_lifetime"]
        elif self.hev_on:
            hev_option = hev_options_reverse["hev"]
        elif self.anti_veto_on:
            hev_option = hev_options_reverse["anti_veto"]
        else:
            hev_option = hev_options_reverse["hev_tag"]

        return hev_option

    def make_channel_map(self):
        """Creates a channel map by replacing the channel name with its current meaning."""
        ng_option = self._infer_ng_channel_meaning()
        hev_option = self._infer_hev_channel_meaning()

        on_ng_ch = self.neutron_generator_channel_options[ng_option]
        on_hev_ch = self.hev_channel_options[hev_option]

        channel_map = {aq_ch.name.lower(): int(aq_ch) for aq_ch in AqmonChannels}

        for kind in ["_start", "_stop"]:
            channel_map[on_hev_ch + kind] = channel_map.pop("hev" + kind)
            channel_map[on_ng_ch + kind] = channel_map.pop("neutron_generator" + kind)

        self.veto_names = ["busy", on_ng_ch, on_hev_ch]
        if self.reconstruct_anti_veto_from_ng:
            self.veto_names.append("anti_veto")
        self.channel_map = channel_map

    @property
    def reconstruct_anti_veto_from_ng(self):
        """If HEV and Anti-Veto are on, we reconstruct the Anti-Veto interval from the NG input.

        HEV and Anti-Veto would othenwise be on the same AQMon channel. In principle we can always
        do this, once we tested that the reconstruction works.

        """
        return self.anti_veto_on and self.hev_on
