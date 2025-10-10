
class PairScannerUtils:
    @staticmethod
    def is_pair_scanner(scanner):
        return hasattr(scanner, 'is_pair_scanner') and scanner.is_pair_scanner