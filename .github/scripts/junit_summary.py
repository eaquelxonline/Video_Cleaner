#!/usr/bin/env python3
"""Firebase Test Lab JUnit XML'inden (test_result_*.xml) özet: test sayıları ve KALDI olan testler.

Kullanım: junit_summary.py test_result_1.xml [...]
Firebase öznitelikleri tek tırnakla yazar; düzenli ifade yerine XML ayrıştırıcı kullanılır.
"""
import sys
import xml.etree.ElementTree as ET


def main():
    for path in sys.argv[1:]:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as error:
            print(f"{path}: okunamadı ({error})")
            continue
        for suite in root.iter("testsuite"):
            print(f"testler: {suite.get('tests')} · kalan: {suite.get('failures')} · hata: {suite.get('errors')} · atlanan: {suite.get('skipped')}")
        for case in root.iter("testcase"):
            failure = case.find("failure")
            if failure is None:
                failure = case.find("error")
            if failure is not None:
                text = (failure.text or "").strip()
                first = text.splitlines()[0][:300] if text else ""
                print(f"KALDI: {case.get('classname')}.{case.get('name')} — {first}")


if __name__ == "__main__":
    main()
